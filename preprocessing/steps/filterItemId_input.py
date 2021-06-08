# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# # Filter Itemid Input
#
# This script is used for filtering itemids from TABLE INPUTEVENTS.
#
# 1. We check number of units of each itemid and choose the major unit as the target of unit conversion.
# 2. In this step we do not apply any filtering to the data.
#
# ## Output
#
# 1. itemid of observations for inputevents.
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
from multiprocessing import Pool
from pathlib import Path

from preprocessing.utils import getConnection


def _stat_inputevents_unit_task(itemid, admission_ids_txt):
    tconn = getConnection()
    tcur = tconn.cursor()
#     tcur.execute('SELECT amountuom, count(amountuom) FROM mimiciii.inputevents_cv \
#                 WHERE amountuom is not null and itemid = '+ str(itemid) +' and hadm_id in ('+admission_ids_txt+') group by amountuom')
#     tcur.execute('select coalesce(amountuom, \'\'), count(*) from (select amountuom, itemid, hadm_id from mimiciii.inputevents_cv union select amountuom, itemid, hadm_id from mimiciii.inputevents_mv) \
#         where itemid={0} and hadm_id in (select hadm_id from admission_ids) group by amountuom'.format(itemid))
    tcur.execute(
        'select amountuom, sum(count::int) from (                    select coalesce(amountuom, \'\') as amountuom, count(*) from mimiciii.inputevents_cv where itemid = {0} and hadm_id in (select * from admission_ids) group by amountuom                    union all                    select coalesce(amountuom, \'\') as amountuom, count(*) from mimiciii.inputevents_mv where itemid = {0} and hadm_id in (select * from admission_ids) group by amountuom                    ) as t where amountuom<>\'\' group by amountuom'.format(itemid))
    outputunits = tcur.fetchall()
    outputunits = sorted(outputunits, key=lambda tup: tup[1])
    outputunits.reverse()
    total = 0
    for o in outputunits:
        total += o[1]
    if(total == 0):
        return (itemid, None, None)
    percentage = float(outputunits[0][1]) / total * 100.0
    tconn.close()
    return (itemid, percentage, outputunits)


def filterItemId_input(args):
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
    valid_input = []
    valid_input_unit = []

    # %%
    # inputevents

    p = Pool(args.num_workers)
    valid_vupairs = [p.apply_async(_stat_inputevents_unit_task, args=(
        i, admission_ids_txt)) for i in input_itemid]
    p.close()
    p.join()
    valid_vupairs = [x.get() for x in valid_vupairs]

    # ## iterate thru each itemID
    # For each item id, we count number of observations for each unit of measurement.
    #
    # For example,
    # IN 225883 : 98.24 : 3 : [('dose', 16477L), ('mg', 251L), ('grams', 44L)]
    # This means that for itemid 225883, there are:
    # 1. 16477 records using dose as its unit of measurement.
    # 2. 251 records using mg as its unit of measurement.
    # 3. 44 records using grams as its unit of measurement.
    #
    # dose has 98.24% over all the observations for this itemid, we can say that dose is a majority unit.
    # 1. We will keep this itemid because 98% is high. we can relatively safe to discard the observations that has different unit of measurement. i.e. if we discard mg and grams, we lose 251+44 records which is little, compared to 16477 records we can keep.
    # 2. We will record main unit of measurement for this itemID as dose.

    # %%
    valid_vupairs = [x for x in valid_vupairs if x[1] is not None]
    valid_vupairs_des = sorted(valid_vupairs, key=lambda x: x[1])

    np.save(cachedir.joinpath('res/filtered_input_raw.npy'),
            {'raw': valid_vupairs})

    # %%
    conn = getConnection()
    sql = 'select hadm_id, amountuom, count(amountuom) from mimiciii.inputevents_cv where itemid={0} group by hadm_id, amountuom union all select hadm_id, amountuom, count(amountuom) from mimiciii.inputevents_mv where itemid={0} group by hadm_id, amountuom order by hadm_id'
    for itemid in [x[0] for x in valid_vupairs_des[:14]]:
        cur = conn.cursor()
        cur.execute(sql.format(itemid))
        results = cur.fetchall()

    # %%
    valid_vupairs = np.load(cachedir.joinpath('res/filtered_input_raw.npy'),
                            allow_pickle=True).tolist()['raw']
    valid_input = [x[0] for x in valid_vupairs]
    valid_input_unit = [x[2][0][0] for x in valid_vupairs]

    np.save(cachedir.joinpath('res/filtered_input.npy'),
            {'id': valid_input, 'unit': valid_input_unit})
