# # Filter Itemid Chart
#
# This script is used for filtering itemids from TABLE CHARTEVENTS.
#
# 1. We check number of units of each itemid and choose the major unit as the target of unit conversion.
# 2. In this step we get 3 kinds of features:
#     - numerical features
#     - categorical features
#     - ratio features, this usually happens in blood pressure measurement, such as "135/70".
#
# ## Output
#
# 1. itemid of observations for chartevents.
# 2. unit of measurement for each itemid.

# %%
from __future__ import print_function

import psycopg2
import datetime
import sys
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
import os.path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count
import re
from tqdm import tqdm as tqdm
from pathlib import Path

from preprocessing.utils import getConnection


def stat_chart_unit_task(ilist, admission_ids_txt):
    subresults = []
    tconn = getConnection()

    for i in tqdm(ilist):
        # for each itemID select number of rows group by unit of measurement.
        tcur = tconn.cursor()
        tcur.execute('SELECT coalesce(valueuom, \'\'), count(*) FROM mimiciii.chartevents WHERE itemid = ' +
                     str(i) + ' and hadm_id in (select * from admission_ids) group by valueuom')
        chartunits = tcur.fetchall()
        chartunits = sorted(chartunits, key=lambda tup: tup[1])
        chartunits.reverse()

        # count number of observation that has non numeric value
        tcur = tconn.cursor()
        tcur.execute('SELECT count(*) FROM mimiciii.chartevents WHERE itemid = ' +
                     str(i) + ' and hadm_id in (select * from admission_ids) and valuenum is null')
        notnum = tcur.fetchone()
        notnum = notnum[0]

        # total number of observation
        tcur = tconn.cursor()
        tcur.execute('SELECT count(*) FROM mimiciii.chartevents WHERE itemid = ' +
                     str(i) + ' and hadm_id in (select * from admission_ids)')
        total = tcur.fetchone()
        total = total[0]

        subresults.append((i, chartunits, notnum, total))

    tconn.close()
    return subresults


def numerical_ratio(units):
    res = list(map(lambda unit: re.match(
        r'(\d+\.\d*)|(\d*\.\d+)|(\d+)', unit), units))
    numerical_ratio = 1.0 * len([1 for r in res if r is not None]) / len(res)
    return numerical_ratio


# %%
def dropped_value_list_unit_task(dropped_id):
    conn = getConnection()
    dropped_value = []
    for d in tqdm(dropped_id):
        cur = conn.cursor()
        cur.execute('SELECT value, valueuom, count(*) as x FROM mimiciii.chartevents as lb                     WHERE itemid = ' +
                    str(d) + ' and hadm_id in (select * from admission_ids) GROUP BY value, valueuom ORDER BY x DESC')
        droped_outs = cur.fetchall()
        drop_array = []
        ct = 0
        total = 0
        for dx in droped_outs:
            total += dx[2]
        units = []
        for dx in droped_outs:
            ct += 1
            if(ct > 20):
                break
            dx = list(dx)
        dropped_value.append((d, droped_outs))
    conn.close()
    return dropped_value


def filterItemId_chart(args):
    conn = getConnection()

    cachedir = Path(args.cachedir)
    _adm = np.load(cachedir.joinpath('res/admission_ids.npy'), allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    db = np.load(cachedir.joinpath('res/itemids.npy'), allow_pickle=True).tolist()
    input_itemid = db['input']
    output_itemid = db['output']
    chart_itemid = db['chart']
    lab_itemid = db['lab']
    microbio_itemid = db['microbio']
    prescript_itemid = db['prescript']

    # %%
    # numworkers = cpu_count() // 2
    numworkers = args.num_workers
    p = Pool(numworkers)
    ilists = np.array_split(chart_itemid, numworkers)
    results = [p.apply_async(stat_chart_unit_task, args=(
        ilist, admission_ids_txt)) for ilist in ilists]
    p.close()
    p.join()
    results = [x.get() for x in results]
    results = itertools.chain.from_iterable(results)
    # results = []
    # for i in tqdm(chart_itemid):
    #     result = stat_chart_unit_task(i, admission_ids_txt)
    #     results.append(result)
    np.save(cachedir.joinpath('res/filtered_chart_raw.npy'), {'raw': results})

    # ## First filtering of categorical features
    #
    # All features with numerical values < 80% of all records are possible categorical features. In this step we drop them for later analyzing.

    # %%
    results = np.load(cachedir.joinpath('res/filtered_chart_raw.npy'),
                      allow_pickle=True).tolist()['raw']
    valid_chart = []
    valid_chart_unit = []
    valid_chart_cate = []
    valid_chart_num = []
    dropped = []
    multiple_units = []
    for x in results:
        i, chartunits, notnum, total = x[0], x[1], x[2], x[3]

        # calculate percentage of the top frequent unit compared to all observation.
        total2 = 0
        unitnum = 0
        for c in chartunits:
            total2 += c[1]
            if c[0] != '':
                unitnum += 1
        if total2 == 0:
            continue
        percentage = float(chartunits[0][1]) / total2 * 100.
        if unitnum > 1:
            multiple_units.append((i, chartunits, percentage))

        # if the percentage of numeric number is less, then dropped it, and make it categorical feature.
        percentage = float(total - notnum)*100 / total
        if(percentage < 80):
            dropped.append(i)
            continue
        valid_chart.append(i)
        valid_chart_unit.append(chartunits[0][0])

    # ## Unit inconsistency
    #
    # Here are itemids having two or more different units.
    #
    # For [211, 505], they have the same unit in fact. Keep them.
    #
    # For [3451, 578, 113], the major unit covers > 90% of all records. Keep them.
    #
    # For [3723], it is just a typo and we keep all.

    # %%
    for i, chartunits, percentage in sorted(multiple_units, key=lambda x: x[2]):
        total2 = sum([t[1] for t in chartunits])
        percentage = float(chartunits[0][1]) / total2 * 100.

    # %%
    dropped_id = dropped

    # %%
    dropped_value = []
    numworkers = 4
    p = Pool(numworkers)
    dropped_id_units = np.array_split(dropped_id, numworkers)
    dropped_value_list = [p.apply_async(dropped_value_list_unit_task, args=(
        dropped_id_unit,)) for dropped_id_unit in dropped_id_units]
    dropped_value_list = [x.get() for x in dropped_value_list]
    dropped_value = list(itertools.chain.from_iterable(dropped_value_list))
    np.save(cachedir.joinpath('res/chart_dropped_value.npy'), dropped_value)

    # %%

    # ## Store selected features in first filtering
    #
    # These features are all numerical features.

    # %%
    np.save(cachedir.joinpath('res/filtered_chart.npy'),
            {'id': valid_chart, 'unit': valid_chart_unit})
    # np.save('res/filtered_chart_cate',{'id':[223758],'unit':None})

    # ## Divide dropped features in first filtering
    #
    # - Features with the ratio of non-numerical values(values that cannot pass the parser) > 0.5: categorical features
    # - Features with the ratio of ratio values > 0.5: ratio features
    # - otherwise: (possible) numerical features, we will parse them later

    # %%
    dropped_value = np.load(cachedir.joinpath('res/chart_dropped_value.npy'),
                            allow_pickle=True).tolist()
    valid_chart_num = []
    valid_chart_num_unit = []
    valid_chart_cate = []
    valid_chart_ratio = []
    for d, droped_outs in dropped_value:
        ascnum = 0
        rationum = 0
        for value, valueuom, count in droped_outs:
            value = str(value)
            isasc = re.search(r'(\d+\.\d*)|(\d*\.\d+)|(\d+)', value) is None
            isratio = re.fullmatch(
                r'{0}\/{0}'.format(r'((\d+\.\d*)|(\d*\.\d+)|(\d+))'), value) is not None
            if isasc:
                ascnum += 1
            if isratio:
                rationum += 1
        if ascnum / len(droped_outs) >= 0.5:
            valid_chart_cate.append(d)
        elif rationum / len(droped_outs) >= 0.5:
            valid_chart_ratio.append(d)
        else:
            valid_chart_num.append(d)
            if droped_outs[0][1] is None:
                valid_chart_num_unit.append('')
            else:
                valid_chart_num_unit.append(droped_outs[0][1])


    # ## Store 3 kinds of features

    # %%
    np.save(cachedir.joinpath('res/filtered_chart_num'),
            {'id': valid_chart_num, 'unit': valid_chart_num_unit})
    np.save(cachedir.joinpath('res/filtered_chart_cate'), {'id': valid_chart_cate, 'unit': None})
    np.save(cachedir.joinpath('res/filtered_chart_ratio'),
            {'id': valid_chart_ratio, 'unit': None})
