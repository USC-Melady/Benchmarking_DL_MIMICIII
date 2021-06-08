# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# # Processing
#
# In this step we generate sparse matrix, general information and ICD9 codes for each patient.

# %%
from __future__ import print_function

from pathlib import Path
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
import traceback
import shutil
import time
from collections import OrderedDict
from tqdm import tqdm

from preprocessing.utils import getConnection
from preprocessing.utils import parseUnitsMap
from preprocessing.utils import parseNum
from preprocessing.utils import sparsify

# %matplotlib inline
# ## Processing inputevents
#
# 1. Discard records without starttime.
# 2. Discard records without amount.
# 3. If the itemid matches manual rules, convert it according to the rule; else only keep the main unit and discard all other records not having the main unit.

# %%


def convert_units(unitmap, src_unit, dst_unit, src_value, f):
    try:
        src_ratio = unitmap['umap'][src_unit]
        dst_ratio = unitmap['umap'][dst_unit]
    except:
        print('converterror: ', unitmap, src_unit,
              dst_unit, src_value, file=f)
        return None
    if src_ratio == 0:
        return None
    else:
        return float(src_value) / src_ratio * dst_ratio


def processing_inputevents(aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index):
    cur = conn.cursor()
    sql = '''select starttime, itemid, amount, amountuom from mimiciii.inputevents_mv where amount>0 and hadm_id={0} and itemid in (select * from mengcztemp_itemids_valid_input)
union all
select charttime as starttime, itemid, amount, amountuom from mimiciii.inputevents_cv where amount>0 and hadm_id={0} and itemid in (select * from mengcztemp_itemids_valid_input)'''.format(aid)
    cur.execute(sql)
    inputevents = cur.fetchall()
    inputevents_wholedata = []
    for ie in inputevents:
        starttime, itemid, amount, amountuom = ie[0], ie[1], ie[2], ie[3]
        # discard values with no starttime
        if starttime is None:
            print('no starttime: ', ie, file=f)
            continue
        # discard values with no amount
        if amount is None:
            print('no amount: ', ie, file=f)
            continue
        # convert units...
        amountuom = amountuom.replace(' ', '').lower()
        unitmap = UNITSMAP['inputevents']
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        if itemid in unitmap.keys():
            dst_value = convert_units(
                unitmap[itemid], amountuom, mainunit, amount, f)
        else:
            if amountuom == mainunit:
                dst_value = float(amount)
            else:
                dst_value = None
        if dst_value is None:
            print('not convertible: ', ie, file=f)
            continue
        inputevents_wholedata.append(
            ['ie', (starttime - admittime).total_seconds(), [starttime, itemid, dst_value, mainunit]])
    return inputevents_wholedata

# processing_inputevents(184834, datetime.datetime.now(), getConnection(), sys.stdout)

# ## Processing outputevents
#
# We only need to discard records without starttime or value.

# %%


def processing_outputevents(aid, admittime, conn, f):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value FROM mimiciii.outputevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_output)')
    outputevents = cur.fetchall()
    wholedata = []
    for oe in outputevents:
        # check date
        if(oe[0] == None):
            print('no starttime', oe, file=f)
            continue
        # discard none value
        if(oe[2] == None):
            print('no value', oe, file=f)
            continue
        # no need to check unit all is mL
        oe = list(oe)
        oe.append('ml')
        wholedata.append(['oe', (oe[0]-admittime).total_seconds(), oe])
    return wholedata

# processing_outputevents(184834, datetime.datetime.now(), getConnection(), f=sys.stdout)

# ## Processing chartevents/labevnets
#
# 1. Discard records without starttime or value/valueuom.
# 2. Process 4 kinds of chartevents/labevents features separately.
#     1. valid numerical features(numerical features not needing parsing): only need to convert units.
#     2. categorical features: only need to map strings to integers.
#     3. possible numerical features(values need parsing):
#         1. parse values
#         2. convert units
#     4. ratio features: store two numbers in the ratio separately.

# %%


def processing_chartevents(aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,valuenum,valueuom FROM mimiciii.chartevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_chart)')
    chartevents = cur.fetchall()
    wholedata = []
    for ce in chartevents:
        # check date
        if ce[0] is None:
            print('no starttime: ', ce, file=f)
            continue

        # discard none value and none valueuom
        if ce[2] is None:
            print('no value: ', ce, file=f)
            continue

        # tuple to list
        ce = list(ce)

        # convert units...
        if ce[3] is None:
            ce[3] = ''
        itemid, valuenum, valueuom = ce[1], ce[2], ce[3]
        valueuom = valueuom.replace(' ', '').lower()
        unitmap = UNITSMAP['chartevents']
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        ce[3] = mainunit
        if itemid in unitmap.keys():
            dst_value = convert_units(
                unitmap[itemid], valueuom, mainunit, valuenum, f)
        else:
            if valueuom == mainunit or valueuom == '':
                dst_value = float(valuenum)
            else:
                dst_value = None
        ce[2] = dst_value

        # discard none value
        if(ce[2] == None):
            print('not convertible: ', ce, file=f)
            continue

        wholedata.append(
            ['ce', (ce[0]-admittime).total_seconds(), list(ce)])
    return wholedata


def processing_chartevents_cate(aid, admittime, conn, f, catedict):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value,valueuom FROM mimiciii.chartevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_chart_cate)')
    chartevents = cur.fetchall()
    wholedata = []
    for ce in chartevents:
        # check date
        if ce[0] is None:
            print('no starttime: ', ce, file=f)
            continue

        # discard none value
        if ce[2] is None:
            print('no value: ', ce, file=f)
            continue

        # tuple to list
        ce = list(ce)

        # map to num
        ce[2] = catedict[ce[1]][ce[2]]
        if ce[2] is None:
            continue
        wholedata.append(
            ['cecate', (ce[0]-admittime).total_seconds(), list(ce)])
    return wholedata


def processing_chartevents_num(aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value,valueuom FROM mimiciii.chartevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_chart_num)')
    chartevents = cur.fetchall()
    wholedata = []
    # parse numerical values
    for ce in chartevents:
        # tuple to list
        ce = list(ce)

        if ce[2] is None:
            print('no value: ', ce, file=f)
            continue
        ce2res = parseNum(ce[2])

        # check date
        if(ce[0] == None):
            print('no starttime: ', ce, file=f)
            continue

        # discard none value
        if(ce2res == None):
            #             writeline(f,'lenum None value :' + str(le))
            print('not parsed: ', ce, file=f)
            continue
        else:
            ce[2] = ce2res

        # check unit
        unitmap = UNITSMAP['chartevents']
        if ce[3] is None:
            ce[3] = ''
        currentunit = ce[3].replace(' ', '').replace(
            '<', '').replace('>', '').replace('=', '').lower()
        mainunit = allitem_unit[map_itemid_index[ce[1]][0]]
        if(currentunit == mainunit or currentunit == ''):
            pass
        else:
            if ce[1] in unitmap.keys():
                ce[2] = convert_units(
                    unitmap[ce[1]], currentunit, mainunit, ce[2], f)
            else:
                if currentunit != mainunit:
                    ce[2] = None

        # discard none value
        if(ce[2] == None):
            print('not convertible: ', ce, file=f)
            continue

        wholedata.append(
            ['cenum', (ce[0]-admittime).total_seconds(), list(ce)])
    return wholedata


def processing_chartevents_ratio(aid, admittime, conn, f):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value,valueuom FROM mimiciii.chartevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_chart_ratio)')
    chartevents = cur.fetchall()
    wholedata = []
    for ce in chartevents:
        ce = list(ce)
        if ce[0] is None:
            print('no starttime: ', ce, file=f)
            continue
        if ce[2] is None:
            print('no value: ', ce, file=f)
            continue
        try:
            fs = ce[2].split('/')
            f1, f2 = fs[0], fs[1]
            if f1 != '':
                ce[2] = float(f1)
                wholedata.append(
                    ['leratio_1', (ce[0] - admittime).total_seconds(), list(ce)])
            if f2 != '':
                ce[2] = float(f2)
                wholedata.append(
                    ['leratio_2', (ce[0] - admittime).total_seconds(), list(ce)])
        except:
            print('not parsed: ', ce, file=f)
            continue
    return wholedata

# processing_chartevents_ratio(136796, datetime.datetime.now(), getConnection(), f=sys.stdout)

# %%


def processing_labevents(aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,valuenum,valueuom FROM mimiciii.labevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_lab)')
    chartevents = cur.fetchall()
    wholedata = []
    for ce in chartevents:
        # check date
        if ce[0] is None:
            print('no starttime: ', ce, file=f)
            continue

        # discard none value
        if ce[2] is None:
            print('no value: ', ce, file=f)
            continue

        # tuple to list
        ce = list(ce)

        # convert units...
        if ce[3] is None:
            ce[3] = ''
        itemid, valuenum, valueuom = ce[1], ce[2], ce[3]
        valueuom = valueuom.replace(' ', '').replace(
            '<', '').replace('>', '').replace('=', '').lower()
        unitmap = UNITSMAP['labevents']
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        ce[3] = mainunit
        if itemid in unitmap.keys():
            dst_value = convert_units(
                unitmap[itemid], valueuom, mainunit, valuenum, f)
        else:
            if valueuom == mainunit or valueuom == '':
                dst_value = float(valuenum)
            else:
                dst_value = None
        ce[2] = dst_value

        # discard none value
        if ce[2] is None:
            print('not convertible: ', ce, file=f)
            continue

        wholedata.append(
            ['le', (ce[0]-admittime).total_seconds(), list(ce)])
    return wholedata


def processing_labevents_cate(aid, admittime, conn, f, catedict):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value,valueuom FROM mimiciii.labevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_lab_cate)')
    labevents = cur.fetchall()
    wholedata = []
    for le in labevents:
        # check date
        if le[0] is None:
            print('no starttime: ', le, file=f)
            continue

        # discard none value
        if le[2] is None:
            print('no value: ', le, file=f)
            continue

        # tuple to list
        le = list(le)

        # map to num
        le[2] = catedict[le[1]][le[2]]
        if le[2] is None:
            continue
        wholedata.append(
            ['lecate', (le[0]-admittime).total_seconds(), list(le)])
    return wholedata


def processing_labevents_num(aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value,valueuom FROM mimiciii.labevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_lab_num)')
    labevents = cur.fetchall()
    wholedata = []
    # parse numerical values
    for le in labevents:
        # tuple to list
        le = list(le)

        if le[0] is None:
            print('no starttime', le, file=f)
            continue

        if le[2] is None:
            print('no value: ', le, file=f)
            continue

        # translate values
        le2res = parseNum(le[2])

        # discard none value
        if le2res is None:
            print('not parsed: ', le, file=f)
            continue
        else:
            le[2] = le2res

        # check unit
        unitmap = UNITSMAP['labevents']
        if le[3] is None:
            le[3] = ''
        currentunit = le[3].replace(' ', '').replace(
            '<', '').replace('>', '').replace('=', '').lower()
        mainunit = allitem_unit[map_itemid_index[le[1]][0]]
        if(currentunit == mainunit or currentunit == ''):
            pass
        else:
            if le[1] in unitmap.keys():
                le[2] = convert_units(
                    unitmap[le[1]], currentunit, mainunit, le[2], f)
            else:
                if currentunit != mainunit:
                    le[2] = None

        # discard none value
        if(le[2] == None):
            print('not convertible: ', le, file=f)
            continue

        wholedata.append(
            ['lenum', (le[0]-admittime).total_seconds(), list(le)])
    return wholedata


def processing_labevents_ratio(aid, admittime, conn, f):
    cur = conn.cursor()
    cur.execute('SELECT charttime,itemid,value,valueuom FROM mimiciii.labevents WHERE hadm_id = ' +
                str(aid)+' and itemid in (select * from mengcztemp_itemids_valid_lab_ratio)')
    chartevents = cur.fetchall()
    wholedata = []
    for ce in chartevents:
        ce = list(ce)
        if ce[0] is None:
            print('no starttime: ', ce, file=f)
        if ce[2] is None:
            print('no value: ', ce, file=f)
            continue
        try:
            fs = ce[2].split('/')
            f1, f2 = fs[0], fs[1]
            if f1 != '':
                ce[2] = float(f1)
                wholedata.append(
                    ['leratio_1', (ce[0] - admittime).total_seconds(), list(ce)])
            if f2 != '':
                ce[2] = float(f2)
                wholedata.append(
                    ['leratio_2', (ce[0] - admittime).total_seconds(), list(ce)])
        except:
            print('not parsed: ', ce, file=f)
            continue
    return wholedata

# processing_labevents_ratio(145834, datetime.datetime.now(), getConnection(), sys.stdout)

# ## Process microbiologyevents
#
# 1. Discard records without starttime.
# 2. Parse dose value in dilution_text. Values which can be parsed only contains '<'/'>'/'=' and numbers.

# %%


def processing_microbiologyevents(aid, admittime, conn, f, valid_microbio):
    wholedata = []
    for m in valid_microbio:
        cur = conn.cursor()
        sql = 'SELECT charttime,(spec_itemid,org_itemid,ab_itemid),dilution_text,\'uom\' FROM mimiciii.microbiologyevents WHERE hadm_id = '+str(aid)
        m = list(map(str, m))
        if(m[0] == 'None'):
            sql += ' and spec_itemid is null'
        else:
            sql += ' and spec_itemid = '+m[0]

        if(m[1] == 'None'):
            sql += ' and org_itemid is null'
        else:
            sql += ' and org_itemid = '+m[1]

        if(m[2] == 'None'):
            sql += ' and ab_itemid is null'
        else:
            sql += ' and ab_itemid = '+m[2]

        cur.execute(sql)
        microevents = cur.fetchall()
        for me in microevents:
            me = list(me)
            for x in range(len(m)):
                try:
                    m[x] = int(m[x])
                except:
                    m[x] = None
            me[1] = tuple(m)
            # checkdate
            if(me[0] == None):
                #                 writeline(f,'me date 0 : '+ " : "+str(me))
                print('no starttime: ', me, file=f)
                continue

            # discard none value
            if(me[2] == None):
                #                 writeline(f,'MICRO None value :' + str(me))
                print('no value: ', me, file=f)
                continue

            # tuple to list
            me = list(me)

            # formatting
            dose = me[2]
            dose = dose.replace('<', '').replace('>', '').replace('=', '')
            numVal = None
            if(dose == ''):
                #                 writeline(f,'me parse fail : '+ " : "+str(me))
                print('not parsed: ', me, file=f)
                continue
            try:
                numVal = float(dose)
            except:
                #                 writeline(f,'me parse fail : '+ " : "+str(me))
                print('not parsed: ', me, file=f)
                continue

            me[2] = numVal

            # discard none value; check again after process
            if(me[2] == None):
                #                 writeline(f,'MICRO None value :' + str(me))
                print('not parsed: ', me, file=f)
                continue

            wholedata.append(
                ['me', (me[0]-admittime).total_seconds(), list(me)])
    return wholedata

# processing_microbiologyevents(149416, datetime.datetime.now(), getConnection(), sys.stdout)

# ## Process prescriptionevents
#
# 1. Discard records without starttime.
# 2. Parse values. Values containing only ','/'<'/'>'/'='/' ' and numbers can be parsed.
# 3. Discard all none values.
# 4. Convert units.

# %%


def processing_prescriptionevents(aid, admittime, conn, f, allitem_unit, map_itemid_index):
    wholedata = []
    cur = conn.cursor()
    cur.execute('SELECT startdate,formulary_drug_cd,dose_val_rx,dose_unit_rx FROM mimiciii.prescriptions WHERE hadm_id = ' +
                str(aid)+' and formulary_drug_cd in (select * from mengcztemp_itemids_valid_prescript)')
    presevents = cur.fetchall()
    for pe in presevents:
        # checkdate
        if(pe[0] == None):
            #             writeline(f,'pe date 0 : '  + " : "+str(pe))
            print('no starttime: ', pe, file=f)
            continue

        # tuple to list
        pe = list(pe)

        if pe[2] is None:
            print('no value: ', pe, file=f)
            continue

        # formatting the value
        dose = pe[2]
        dose = dose.replace(',', '').replace('<', '').replace(
            '>', '').replace('=', '').replace(' ', '')
        numVal = None
        try:
            numVal = float(dose)
        except:
            if(len(dose.split('-')) == 2):
                strs = dose.split('-')
                try:
                    numVal = (float(strs[0]) + float(strs[1]))/2.0
                except:
                    #                     writeline(f,'pe parse fail : '  + " : "+str(pe))
                    print('not parsed: ', pe, file=f)
                    continue
            else:
                #                 writeline(f,'pe parse fail : '  + " : "+str(pe))
                print('not parsed: ', pe, file=f)
                continue

        pe[2] = numVal

        # discard none value
        if(pe[2] == None):
            #             writeline(f,'PRES None value :' + str(pe))
            print('not parsed: ', pe, file=f)
            continue

        # check unit
        # convert units...
        if pe[3] is None:
            pe[3] = ''
        itemid, valuenum, valueuom = pe[1], pe[2], pe[3]
        valueuom = valueuom.replace(' ', '').lower()
        mainunit = allitem_unit[map_itemid_index[itemid][0]]
        if valueuom == mainunit or valueuom == '':
            dst_value = float(valuenum)
        else:
            dst_value = None
        pe[2] = dst_value

        # discard none value
        if(pe[2] == None):
            #             writeline(f, 'PRES None value :' + str(pe))
            print('not convertible: ', pe, file=f)
            continue

        wholedata.append(
            ['pe', (pe[0]-admittime).total_seconds(), list(pe)])
    return wholedata

# ## Processing
#
# In this step, we generate the matrix of time series for each admission.
#
# 1. Discard admissoins without admittime.
# 2. Collect records from inputevents, outputevents, chartevents, labevents, microbiologyevents and prescriptionevents.
# 3. For possible conflictions(different values of the same feature occur at the same time):
#     1. For numerical values:
#         1. For inputevents/outputevents/prescriptions, we use the sum of all conflicting values.
#         2. For labevents/chartevents/microbiologyevents, we use the mean of all conflicting values.
#     2. For categorical values: we use the value appear first and record that confliction event in the log.
#     3. For ratio values: we separate the ratio to two numbers and use the mean of each of them.

# %%
# integrate the time series array for a single patient


def processing_func(aid, f, UNITSMAP, allitem_unit, map_itemid_index, catedict, valid_microbio, allids):
    conn = getConnection()

    # get admittime
    cur = conn.cursor()
    cur.execute(
        'select admittime from mimiciii.admissions where hadm_id={0}'.format(aid))
    admission = cur.fetchone()
    if admission is None:
        return None
    admittime = admission[0]
    if admittime is None:
        return None
    wholedata = []

    # preprocess inputevents
    wholedata.append(processing_inputevents(
        aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index))

    # preprocess outputevents
    wholedata.append(processing_outputevents(aid, admittime, conn, f))

    # preprocess chartevents
    wholedata.append(processing_chartevents(
        aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index))
    wholedata.append(processing_chartevents_cate(
        aid, admittime, conn, f, catedict))
    wholedata.append(processing_chartevents_num(
        aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index))
    wholedata.append(processing_chartevents_ratio(aid, admittime, conn, f))

    # preprocess labevents
    wholedata.append(processing_labevents(aid, admittime, conn,
                                          f, UNITSMAP, allitem_unit, map_itemid_index))
    wholedata.append(processing_labevents_cate(
        aid, admittime, conn, f, catedict))
    wholedata.append(processing_labevents_num(
        aid, admittime, conn, f, UNITSMAP, allitem_unit, map_itemid_index))
    wholedata.append(processing_labevents_ratio(aid, admittime, conn, f))

    # preprocess microbiologyevents
    wholedata.append(processing_microbiologyevents(
        aid, admittime, conn, f, valid_microbio))

    # preprocess prescriptionevents
    wholedata.append(processing_prescriptionevents(
        aid, admittime, conn, f, allitem_unit, map_itemid_index))

    # here is the sparse matrix, order by timestamp
    wholedata = sorted(
        list(itertools.chain(*wholedata)), key=itemgetter(1))

    # transform sparse matrix to matrix
    D = len(allids) + 2

    # map time to row
    map_time_index = {}
    index = 0
    for wd in wholedata:
        if(wd[1] not in map_time_index):
            map_time_index[wd[1]] = index
            index += 1

    patient = [[None for i in range(D)]
               for j in range(len(map_time_index))]
    numtodivide = [[0 for i in range(D-2)]
                   for j in range(len(map_time_index))]
#     writeline(f,'len(wholedata) = '+str(len(wholedata)))
#     writeline(f, 'D = '+str(D))
#     writeline(f,'len(patient) = '+str(len(patient)) +' timesteps')

    for wd in wholedata:

        assert patient[map_time_index[wd[1]]][D -
                                              2] == None or patient[map_time_index[wd[1]]][D-2] == wd[1]
        patient[map_time_index[wd[1]]][D-2] = wd[1]
        patient[map_time_index[wd[1]]][D-1] = aid

        if(wd[0] == 'ie' or wd[0] == 'oe' or wd[0] == 'pe'):
            if(patient[map_time_index[wd[1]]][map_itemid_index[wd[2][1]][0]] == None):
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][0]] = wd[2][2]
            else:
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][0]] += wd[2][2]

        if(wd[0] == 'le' or wd[0] == 'ce' or wd[0] == 'me' or wd[0] == 'lenum' or wd[0] == 'cenum'):
            if wd[2][2] is None:
                print('None value: ', wd, file=f)
            if(patient[map_time_index[wd[1]]][map_itemid_index[wd[2][1]][0]] == None):
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][0]] = wd[2][2]
                numtodivide[map_time_index[wd[1]]
                            ][map_itemid_index[wd[2][1]][0]] = 1
            else:
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][0]] += wd[2][2]
                numtodivide[map_time_index[wd[1]]
                            ][map_itemid_index[wd[2][1]][0]] += 1

        if (wd[0].startswith('ceratio') or wd[0].startswith('leratio')):
            ot = int(wd[0].split('_')[1]) - 1
            if wd[2][2] is None:
                print(wd, file=f)
            if(patient[map_time_index[wd[1]]][map_itemid_index[wd[2][1]][ot]] == None):
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][ot]] = wd[2][2]
                numtodivide[map_time_index[wd[1]]
                            ][map_itemid_index[wd[2][1]][ot]] = 1
            else:
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][ot]] += wd[2][2]
                numtodivide[map_time_index[wd[1]]
                            ][map_itemid_index[wd[2][1]][ot]] += 1

        if(wd[0] == 'cecate' or wd[0] == 'lecate'):
            if(patient[map_time_index[wd[1]]][map_itemid_index[wd[2][1]][0]] == None):
                patient[map_time_index[wd[1]]
                        ][map_itemid_index[wd[2][1]][0]] = wd[2][2]
            else:
                print('DUPLICATED :', wd, file=f)

    for i in range(len(map_time_index)):
        for j in range(D-2):
            if(numtodivide[i][j] == 0):
                continue
            try:
                patient[i][j] /= numtodivide[i][j]
            except:
                print('div error: ', i, j, file=f)
    conn.close()
    return patient


def ageLosMortality(aid, f, mapping, cate):
    conn = getConnection()

    cur = conn.cursor()
    cur.execute('SELECT hadm_id,subject_id,admittime,dischtime,deathtime,admission_type,admission_location,insurance,language,religion,marital_status,ethnicity FROM mimiciii.ADMISSIONS WHERE hadm_id='+str(aid))
    admission = cur.fetchone()

    assert admission != None

    subject_id = admission[1]
    admittime = admission[2]
    dischtime = admission[3]
    deathtime = admission[4]

    cur = conn.cursor()
    cur.execute(
        'SELECT dob, dod FROM mimiciii.PATIENTS WHERE subject_id='+str(subject_id))
    patient = cur.fetchone()

    assert patient != None
    birthdate = patient[0]
    final_deathdate = patient[1]
    mortal = 0
    labelGuarantee = 0
    die24 = 0
    die24_48 = 0
    die48_72 = 0
    die30days = 0
    die1year = 0
    if(deathtime != None):
        mortal = 1
        if(deathtime != dischtime):
            labelGuarantee = 1
        secnum = (deathtime - admittime).total_seconds()
        if secnum <= 24 * 60 * 60:
            die24 = 1
        if secnum <= 48 * 60 * 60:
            die24_48 = 1
        if secnum <= 72 * 60 * 60:
            die48_72 = 1
    if dischtime is not None and final_deathdate is not None:
        dischsecnum = (final_deathdate - dischtime).total_seconds()
        if dischsecnum <= 30 * 24 * 60 * 60:
            die30days = 1
        if dischsecnum <= 365 * 24 * 60 * 60:
            die1year = 1

    cur.execute(
        'select curr_service from mimiciii.services where hadm_id='+str(aid))
    curr_service = cur.fetchone()
    if curr_service:
        curr_service = curr_service[0]
    else:
        curr_service = 'NB'

    data = [aid, subject_id, (admittime - birthdate).total_seconds()/(3600*24), (dischtime-admittime).total_seconds(
    )//60., mortal, labelGuarantee, die24, die24_48, die48_72, die30days, die1year, mapping['curr_service'][curr_service]]
    for i in range(5, 12):
        data.append(mapping[cate[i-5]][admission[i]])
    conn.close()
    return data

# ageLosMortality(128652, sys.stdout)

# ## Generate ICD9 codes
#
# Here we convert icd9 codes to category numbers.

# %%


def ICD9(aid, f):
    conn = getConnection()
    cate20 = 0

    cur = conn.cursor()
    cur.execute('SELECT icd9_code FROM mimiciii.DIAGNOSES_ICD WHERE hadm_id=' +
                str(aid)+' ORDER BY seq_num')
    icd9s = cur.fetchall()
    list_icd9 = []
    for icd9 in icd9s:
        icd = icd9[0]
        if icd is None:
            continue
        if(icd[0] == 'V'):
            label_name = 19
            numstr = icd[0:3]+'.'+icd[3:len(icd)]
        elif(icd[0] == 'E'):
            cate20 += 1
            label_name = 20
            numstr = icd
        else:
            num = float(icd[0:3])
            numstr = icd[0:3]+'.'+icd[3:len(icd)]
            if(num >= 1 and num <= 139):
                label_name = 0
            if(num >= 140 and num <= 239):
                label_name = 1
            if(num >= 240 and num <= 279):
                label_name = 2
            if(num >= 280 and num <= 289):
                label_name = 3
            if(num >= 290 and num <= 319):
                label_name = 4
            if(num >= 320 and num <= 389):
                label_name = 5
            if(num >= 390 and num <= 459):
                label_name = 6
            if(num >= 460 and num <= 519):
                label_name = 7
            if(num >= 520 and num <= 579):
                label_name = 8
            if(num >= 580 and num <= 629):
                label_name = 9
            if(num >= 630 and num <= 677):
                label_name = 10
            if(num >= 680 and num <= 709):
                label_name = 11
            if(num >= 710 and num <= 739):
                label_name = 12
            if(num >= 740 and num <= 759):
                label_name = 13
            if(num >= 760 and num <= 779):
                label_name = 14
            if(num >= 780 and num <= 789):
                label_name = 15
            if(num >= 790 and num <= 796):
                label_name = 16
            if(num >= 797 and num <= 799):
                label_name = 17
            if(num >= 800 and num <= 999):
                label_name = 18
        list_icd9.append([aid, icd, numstr, label_name])
    conn.close()
    return list_icd9


def process_patient(aid, args, mapping, cate, UNITSMAP, allitem_unit, map_itemid_index, catedict, valid_microbio, allids):
    cachedir = Path(args.cachedir)
    with open(cachedir.joinpath('admdata/log/adm-{0}.log'.format(str('%.6d' % aid))), 'w') as f:
        try:
            # start_ts = time.time()
            proc = processing_func(
                aid, f, UNITSMAP, allitem_unit, map_itemid_index, catedict, valid_microbio, allids)
            if len(proc) == 0:
                return
            res = {
                'timeseries': sparsify(proc),
                'general': ageLosMortality(aid, f, mapping, cate),
                'icd9': ICD9(aid, f)
            }
            np.save(cachedir.joinpath(
                'admdata/adm-' + str('%.6d' % aid)), res)
            # print('finished {:d} in {:.3f} seconds!'.format(
            #     aid, time.time() - start_ts))
        except Exception as e:
            with open(cachedir.joinpath('admdata/log/admerror-{0}.log'.format(str('%.6d' % aid))), 'w') as ferr:
                traceback.print_exc(file=ferr)
            traceback.print_exc(sys.stdout)
            print('failed at {0}!'.format(aid))


def process_patient_list(aid_list, *args, **kwargs):
    for aid in tqdm(aid_list):
        process_patient(aid, *args, **kwargs)


def add_mortality_labels(aid, args, mapping, cate):
    cachedir = Path(args.cachedir)
    with open(cachedir.joinpath('admdata/log/adm-addml{0}.log'.format(str('%.6d' % aid))), 'w') as f:
        try:
            res = np.load(cachedir.joinpath(
                'admdata/adm-' + str('%.6d' % aid)+'.npy'), allow_pickle=True).tolist()
        except Exception as e:
            traceback.print_exc(file=f)
            return
        res['general'] = ageLosMortality(aid, f, mapping, cate)
        np.save(cachedir.joinpath(
            'admdata/adm-' + str('%.6d' % aid)+'.npy'), res)


def add_mortality_labels_list(aid_list, *args, **kwargs):
    for aid in tqdm(aid_list):
        add_mortality_labels(aid, *args, **kwargs)

# ## Set unit conversion map
#
# We manually set rules for unit conversion and store them in file './config/unitsmap.unit'. The file format is as following:
#
# ```
# tablename:[name of table in database]
# [itemid1],[target unit1],[unit11:ratio11],[unit12:ratio12],...,[unit1n:ratio1n]
# ...
# [itemidn],[target unitn],[unitn1:ration1],[unitn2:ration2],...,[unitnn:rationn]
# ```
#
# The ratio is set using the following rule: $\mathrm{unit1}*\mathrm{ratio1}=\mathrm{unit2}*\mathrm{ratio2}=...=\mathrm{unitn}*\mathrm{ration}$. For example, one row in the file could be: `227114,mg,mg:1,mcg:1000`. It means that $1\mathrm{mg}=1000\mathrm{mcg}$.

# %%


def processing(args):
    UNITSMAP = parseUnitsMap()

    cachedir = Path(args.cachedir)

    # # Set indices for chartevents table
    #
    # We need to add hadm_id as indices to speed up the query. By default it is not added. Thanks for the help from Weijing Tang@UMich!
    #
    # You might need to run `grant postgres to <your username>;` before building indices. https://stackoverflow.com/questions/28584640/postgresql-error-must-be-owner-of-relation-when-changing-a-owner-object/28586288

    # %%
    conn = getConnection()
    cur = conn.cursor()
    # add index to the whole chartevents
    # indicescomm = '''DROP INDEX IF EXISTS chartevents_idx02;
    # CREATE INDEX chartevents_idx02 ON mimiciii.chartevents (hadm_id);'''
    indicescomm = 'CREATE INDEX IF NOT EXISTS chartevents_idx02 ON mimiciii.chartevents (hadm_id);'
    cur.execute(indicescomm)
    conn.commit()

    # %%
    _adm = np.load(cachedir.joinpath('res/admission_ids.npy'),
                   allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    _adm_first = np.load(cachedir.joinpath(
        'res/admission_first_ids.npy'), allow_pickle=True).tolist()
    admission_first_ids = _adm['admission_ids']
    admission_first_ids_set = set(admission_first_ids)

    # %%
    v = np.load(cachedir.joinpath('res/filtered_input.npy'),
                allow_pickle=True).tolist()
    valid_input = v['id']
    valid_input_unit = v['unit']

    v = np.load(cachedir.joinpath('res/filtered_output.npy'),
                allow_pickle=True).tolist()
    valid_output = v['id']

    v = np.load(cachedir.joinpath('res/filtered_chart.npy'),
                allow_pickle=True).tolist()
    valid_chart = v['id']
    valid_chart_unit = v['unit']

    v = np.load(cachedir.joinpath('res/filtered_chart_num.npy'),
                allow_pickle=True).tolist()
    valid_chart_num = v['id']
    valid_chart_num_unit = v['unit']

    v = np.load(cachedir.joinpath('res/filtered_chart_cate.npy'),
                allow_pickle=True).tolist()
    valid_chart_cate = v['id']

    v = np.load(cachedir.joinpath('res/filtered_chart_ratio.npy'),
                allow_pickle=True).tolist()
    valid_chart_ratio = v['id']

    v = np.load(cachedir.joinpath('res/filtered_lab.npy'),
                allow_pickle=True).tolist()
    valid_lab = v['id']
    valid_lab_unit = v['unit']

    v = np.load(cachedir.joinpath('res/filtered_lab_num.npy'),
                allow_pickle=True).tolist()
    valid_lab_num = v['id']
    valid_lab_num_unit = v['unit']

    v = np.load(cachedir.joinpath('res/filtered_lab_cate.npy'),
                allow_pickle=True).tolist()
    valid_lab_cate = v['id']

    v = np.load(cachedir.joinpath('res/filtered_lab_ratio.npy'),
                allow_pickle=True).tolist()
    valid_lab_ratio = v['id']

    v = np.load(cachedir.joinpath('res/filtered_microbio.npy'),
                allow_pickle=True).tolist()
    valid_microbio = v['id']

    v = np.load(cachedir.joinpath('res/filtered_prescript.npy'),
                allow_pickle=True).tolist()
    valid_prescript = v['id']
    valid_prescript_unit = v['unit']

    allids = valid_input+valid_output+valid_chart+valid_chart_num+valid_chart_cate+valid_chart_ratio+valid_chart_ratio + \
        valid_lab+valid_lab_num+valid_lab_cate+valid_lab_ratio + \
        valid_lab_ratio+valid_microbio+valid_prescript
    # print(len(allids), len(set(allids)))

    # ## Create temporary tables for accelerating the query

    # %%
    # put valid ids into database
    conn = getConnection()
    cur = conn.cursor()
    for itemidlist, itemidlistname in zip([valid_input, valid_output, valid_chart, valid_chart_num, valid_chart_cate, valid_chart_ratio, valid_lab, valid_lab_num, valid_lab_cate, valid_lab_ratio], 'valid_input, valid_output, valid_chart, valid_chart_num, valid_chart_cate, valid_chart_ratio, valid_lab, valid_lab_num, valid_lab_cate, valid_lab_ratio'.replace(' ', '').split(',')):
        sql = 'drop table if exists mengcztemp_itemids_{0}'.format(
            itemidlistname)
        cur.execute(sql)
        conn.commit()
        sql = 'create table if not exists mengcztemp_itemids_{0} (    itemid serial PRIMARY KEY     )'.format(
            itemidlistname)
        cur.execute(sql)
        conn.commit()
        for itemid in itemidlist:
            sql = 'insert into mengcztemp_itemids_{0} (itemid) values ({1})'.format(
                itemidlistname, itemid)
            cur.execute(sql)
        conn.commit()
        sql = 'select * from mengcztemp_itemids_{0} limit 100'.format(
            itemidlistname)
        cur.execute(sql)
        res = cur.fetchall()
    #     print(res)

    # %%
    cur = conn.cursor()
    sql = 'drop table if exists mengcztemp_itemids_{0}'.format(
        'valid_prescript')
    cur.execute(sql)
    conn.commit()
    sql = 'create table if not exists mengcztemp_itemids_{0} (    itemid varchar(255) PRIMARY KEY     )'.format(
        'valid_prescript')
    cur.execute(sql)
    conn.commit()
    for itemid in valid_prescript:
        sql = 'insert into mengcztemp_itemids_{0} (itemid) values (\'{1}\')'.format(
            'valid_prescript', itemid)
        cur.execute(sql)
    conn.commit()
    sql = 'select * from mengcztemp_itemids_{0} limit 100'.format(
        'valid_prescript')
    cur.execute(sql)
    res = cur.fetchall()
    # print(res, len(res), len(valid_prescript))

    # # %%
    # print('len(valid_input) = ' + str(len(valid_input)))
    # print('len(valid_output) = ' + str(len(valid_output)))
    # print('len(valid_chart) = ' + str(len(valid_chart)))
    # print('len(valid_chart_num) = ' + str(len(valid_chart_num)))
    # print('len(valid_chart_cate) = ' + str(len(valid_chart_cate)))
    # print('len(valid_chart_ratio) = ' + str(len(valid_chart_ratio)))
    # print('len(valid_lab) = ' + str(len(valid_lab)))
    # print('len(valid_lab_num) = ' + str(len(valid_lab_num)))
    # print('len(valid_lab_cate) = ' + str(len(valid_lab_cate)))
    # print('len(valid_lab_ratio) = ' + str(len(valid_lab_ratio)))
    # print('len(valid_microbio) = ' + str(len(valid_microbio)))
    # print('len(valid_prescript) = ' + str(len(valid_prescript)))
    # print('\nlen(allids) = ' + str(len(allids)))

    # %%
    # map itemids to [0..n] column
    index = 0
    map_itemid_index = {}
    allitem = allids
    allitem_unit = valid_input_unit + ['NOCHECK'] * len(valid_output) + valid_chart_unit + valid_chart_num_unit + ['NOCHECK'] * len(valid_chart_cate) + ['NOCHECK'] * 2 * len(
        valid_chart_ratio) + valid_lab_unit + valid_lab_num_unit + ['NOCHECK']*len(valid_lab_cate) + ['NOCHECK'] * 2 * len(valid_lab_ratio)+['NOCHECK']*len(valid_microbio) + valid_prescript_unit
    for i in range(len(allitem_unit)):
        allitem_unit[i] = allitem_unit[i].replace(' ', '').lower()
    assert len(allitem) == len(allitem_unit)
    for ai in allitem:
        if ai not in map_itemid_index.keys():
            map_itemid_index[ai] = [index]
        else:
            map_itemid_index[ai].append(index)
        index += 1
    # print(map_itemid_index)
    # print(len(map_itemid_index))
    np.save(cachedir.joinpath('res/map_itemid_index.npy'), map_itemid_index)

    # ## Map strings in categorical features to integers and store them to a file

    # %%
    catedict = {}

    if not cachedir.joinpath('res/catedict.npy').exists():
        for i in tqdm(valid_chart_cate):
            cur = conn.cursor()
            cur.execute('SELECT distinct value FROM mimiciii.chartevents WHERE itemid = ' +
                        str(i) + ' and hadm_id in (select * from admission_ids)')
            distinctval = cur.fetchall()
            mapping = {}
            ct = 1
            for d in distinctval:
                mapping[d[0]] = ct
                ct += 1
            catedict[i] = mapping
            # print(i)

        for i in tqdm(valid_lab_cate):
            cur = conn.cursor()
            cur.execute('SELECT distinct value FROM mimiciii.labevents WHERE itemid = ' +
                        str(i) + ' and hadm_id in (select * from admission_ids)')
            distinctval = cur.fetchall()
            mapping = {}
            ct = 1
            for d in distinctval:
                mapping[d[0]] = ct
                ct += 1
            catedict[i] = mapping
            # print(i)

        np.save(cachedir.joinpath('res/catedict.npy'), catedict)
    # print('saved!')

    # %%
    catedict = np.load(cachedir.joinpath(
        'res/catedict.npy'), allow_pickle=True).tolist()
    # print(catedict)

    # %%

    # ## Generate information of patient
    #
    # Here we collect information of one patient, containing its admission_type, admission_location, insurance, language, religion, marital_status and ethnicity.
    #
    # Since all of them are categorical features, we map the strings of each feature to integers and store the mapping.

    # %%
    # generate general information of patient

    # generate map for categorical values
    conn = getConnection()
    cate = ['admission_type', 'admission_location', 'insurance',
            'language', 'religion', 'marital_status', 'ethnicity']
    mapping = {}
    for c in cate:
        cur = conn.cursor()
        cur.execute('select distinct '+c+' from mimiciii.admissions')
        types = cur.fetchall()

        catemapping = {}
        for i in range(len(types)):
            catemapping[types[i][0]] = i
        mapping[c] = catemapping

    # add map for services
    cur = conn.cursor()
    cur.execute('select distinct ' + 'curr_service' +
                ' from mimiciii.services')
    types = cur.fetchall()

    catemapping = {}
    for i, typen in enumerate(types):
        catemapping[typen[0]] = i
    mapping['curr_service'] = catemapping
    # mapping
    mapping['curr_service']
    np.save(cachedir.joinpath('res/adm_catemappings.npy'), mapping)

    # ## Generate non-temporal features
    #
    # Here we collect all non-temporal features only related to the admissions:
    # 1. admission id
    # 2. subject id(for finding the patient of one admission)
    # 3. age(at admittime, unit is day)
    # 4. length of stay(unit is minute)
    # 5. in-hospital mortality label
    # 6. labelGurantee label
    # 7. 1-day mortality(from admittime)
    # 8. 2-day mortality(from admittime)
    # 9. 3-day mortality(from admittime)
    # 10. 30-day mortality(from dischtime)
    # 11. 1-year mortality(from dischtime)
    # 12. admission_type
    # 13. admission_location
    # 14. insurance
    # 15. language
    # 16. religion
    # 17. marital_status
    # 18. ethnicity
    #
    # **Mortality label here is not used, please refer to 8_collect_time_labels.ipynb to get correct mortality labels. We leave them here only for compatibility.**

    # %%

    # ICD9(185777, sys.stdout)

    # %%
    admdata_log_dir = cachedir.joinpath('admdata', 'log')
    admdata_log_dir.mkdir(parents=True, exist_ok=True)

    # ## Save one file for each admission
    #
    # For each admission, we save a separate file for it, which contains:
    # 1. 'timeseries': matrix of time series in form of sparse matrix
    # 2. 'general': non-temporal features
    # 3. 'icd9': list of icd9 category codes


    # %%
    p = Pool(args.num_workers)
    for aid_list in np.array_split(admission_ids, args.num_workers):
        p.apply_async(process_patient_list, args=(aid_list, args, mapping, cate, UNITSMAP,
                                             allitem_unit, map_itemid_index, catedict, valid_microbio, allids))
    p.close()
    p.join()

    # %%
    # add labels about mortality, now we have 1day|2days|3days|in-hospitial|30days|1year

    p = Pool(args.num_workers)
    for aid_list in np.array_split(admission_ids, args.num_workers):
        p.apply_async(add_mortality_labels_list, args=(aid_list, args, mapping, cate))
    p.close()
    p.join()
    # t = 0
    # for aid in admission_ids:
    #     add_mortality_labels(aid, sys.stdout)
    #     t += 1
    #     if t % 100 == 0:
    #         print(t)

    # %%
