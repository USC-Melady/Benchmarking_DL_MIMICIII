# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# # Filter Itemid Lab
#
# This script is used for filtering itemids from TABLE LABEVENTS.
#
# 1. We check number of units of each itemid and choose the major unit as the target of unit conversion.
# 2. In this step we get 3 kinds of features:
#     - numerical features
#     - categorical features
#     - ratio features, this usually happens in blood pressure measurement, such as "135/70".
#
# ## Output
#
# 1. itemid of observations for labevents.
# 2. unit of measurement for each itemid.

# %%
from __future__ import print_function

import psycopg2
from pathlib import Path
import datetime
import sys
import re
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
import os.path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, cpu_count

from preprocessing.utils import getConnection


def stat_lab_unit_task(i, admission_ids_txt):
    conn = getConnection()
    cur = conn.cursor()
    cur.execute('SELECT coalesce(valueuom,\'\'), count(*) FROM mimiciii.labevents WHERE itemid = ' +
                str(i) + ' and hadm_id in (select * from admission_ids) group by valueuom')
    outputunits = cur.fetchall()
    outputunits = sorted(outputunits, key=lambda tup: tup[1])
    outputunits.reverse()

    cur = conn.cursor()
    cur.execute('SELECT count(*) FROM mimiciii.labevents WHERE itemid = ' +
                str(i) + ' and hadm_id in (select * from admission_ids) and valuenum is null')
    notnum = cur.fetchone()

    cur = conn.cursor()
    cur.execute('SELECT count(*) FROM mimiciii.labevents WHERE itemid = ' + str(i) +
                ' and hadm_id in (select * from admission_ids) and valuenum is not null')
    total = cur.fetchone()

    return (i, outputunits, notnum, total)


def filterItemId_lab(args):
    conn = getConnection()

    cachedir = Path(args.cachedir)
    _adm = np.load(cachedir.joinpath('res/admission_ids.npy'),
                   allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    db = np.load(cachedir.joinpath('res/itemids.npy'),
                 allow_pickle=True).tolist()
    input_itemid = db['input']
    output_itemid = db['output']
    chart_itemid = db['chart']
    lab_itemid = db['lab']
    microbio_itemid = db['microbio']
    prescript_itemid = db['prescript']

    # %%

    p = Pool(args.num_workers)
    results = [p.apply_async(stat_lab_unit_task, args=(
        i, admission_ids_txt)) for i in lab_itemid]
    results = [x.get() for x in results]
    np.save(cachedir.joinpath('res/filtered_lab_raw.npy'), {'raw': results})

    # %%
    valid_lab = []
    valid_lab_unit = []
    dropped_id = []
    multiple_units = []
    results = np.load(cachedir.joinpath(
        'res/filtered_lab_raw.npy'), allow_pickle=True).tolist()['raw']
    for x in results:
        i, outputunits, notnum, total = x[0], x[1], x[2], x[3]
        totaltemp = 0
        unitnum = 0
        for o in outputunits:
            totaltemp += o[1]
            if o[0] is not '':
                unitnum += 1
        if(totaltemp == 0):
            continue
        percentage = float(outputunits[0][1]) / totaltemp * 100.

        if unitnum > 1:
            multiple_units.append((i, percentage, totaltemp, outputunits))

        percentage = float(total[0])*100 / (notnum[0]+total[0])

        if(percentage < 95):
            dropped_id.append(i)
            continue

        valid_lab.append(i)
        valid_lab_unit.append(outputunits[0][0])

    # %%
    np.save(cachedir.joinpath('res/filtered_lab.npy'),
            {'id': valid_lab, 'unit': valid_lab_unit})

    # All the units are convertible, so keep all of them.

    # %%

    # %%
    # valid_lab_num = [51475,50845,51280,50935,51479,50922,51501,50925,50856,50981,51213,50915,51497,51046,50835,51176,51180,51194,51196,51130,51131,51132,50906,51076,51422,51517,51229,50946,51471,50899,51515,51369,50992,50958,50926,50961,51228,50877,51494,50990,50991,51489,51488,51061,51225,50894,50989,51209,51516,51493,51476,50911,51003,51482]
    # valid_lab_cate = [51519,51461,51495,51390,51096,51403,51391,51189,51405,50901,51407,51089,51153,51171,51220,51179,51468,51016,51401,51207,51472,51394,51291,51500,51161,51286,51410,51417,51389,51195,51322,51201,51142,51311,51135,51329,51396,51318,51421,50875,51485,51316,51308,51056,51537,51414,51304,51157,51079,51075,51071,51074,51092,51090,51512,50828,50933,51266,51246,51267,51137,51252,51268,51233,50955,50887,51523,51462,50979,51260,50919,51287,51296,51151,51474,51107,51103,51236,50940,50943,50941,51145,51294,50942,51240,51292,51518,50873,51505,51469,50975,51424,51134,51411,50944,50937,51197,51425,51426,51098,51243,51373,51147,51085,51216,51400,51388,51412,50872,51150,51423,51402,50938,50939,51234,51420,51338,51325,51183,51164,50948,51313,50857,51399,51239,51238,51416,51319,51230,51337,51152,51168,51184,51341,51340,51198,51326,51303,51315,50876,51261,51499,50871,51086,51192,51167,51332,51314,51342,51231,51321,51264,51374,51370,51372,51307,51235,51215,51317,51503,51335,51172,50874,51219,51305,51310,51323,51334,51320,51178,51156,51309,51306,51155,51324,51177,51158,51328,50913,51336,51159,51333,51154,51331,51262,51339,51182,51217,50918,51193,51460,51191,51510,51091,51465,51017,51160,51408,51418,51190,51187,51413,50932,51511,51350,51393,51481,51392,51490,51397,51470,51247,51312,51395,51295,51173,51502,51415,51508,51506,51466,51486,51464,51487,51463,50920,50800,50880,50879,50999,50812]
    # valid_lab_set = set(valid_lab_num + valid_lab_cate)
    # leftids = [d for d in dropped_id if d not in valid_lab_set]

    dropped_value = []

    for d in dropped_id:
        # for d in leftids:
        cur = conn.cursor()
        cur.execute('SELECT value, valueuom, count(*) as x FROM mimiciii.labevents as lb                 WHERE itemid = ' +
                    str(d) + ' and hadm_id in (select * from admission_ids) GROUP BY value, valueuom ORDER BY x DESC')
        droped_outs = cur.fetchall()
        drop_array = []
        ct = 0
        total = 0
        for dx in droped_outs:
            total += dx[2]
        for dx in droped_outs:
            ct += 1
            if(ct > 20):
                break
            dx = list(dx)
        dropped_value.append((d, droped_outs))

    np.save(cachedir.joinpath('res/lab_dropped_value.npy'), dropped_value)

    # %%
    dropped_value = np.load(cachedir.joinpath(
        'res/lab_dropped_value.npy'), allow_pickle=True).tolist()
    valid_lab_num = []
    valid_lab_num_unit = []
    valid_lab_cate = []
    valid_lab_ratio = []
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
            valid_lab_cate.append(d)
        elif rationum / len(droped_outs) >= 0.5:
            valid_lab_ratio.append(d)
        else:
            valid_lab_num.append(d)
            if droped_outs[0][1] is None:
                valid_lab_num_unit.append('')
            else:
                valid_lab_num_unit.append(droped_outs[0][1])

    # %%
    # Here we directly do some manual selection to get the list of valid_lab_num and valid_lab_cat

    # %%

    # %%
    np.save(cachedir.joinpath('res/filtered_lab_num'),
            {'id': valid_lab_num, 'unit': valid_lab_num_unit})
    np.save(cachedir.joinpath('res/filtered_lab_cate'),
            {'id': valid_lab_cate, 'unit': None})
    np.save(cachedir.joinpath('res/filtered_lab_ratio'),
            {'id': valid_lab_ratio, 'unit': None})
