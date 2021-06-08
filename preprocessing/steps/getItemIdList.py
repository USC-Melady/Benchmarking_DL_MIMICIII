# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# # Get Itemid List
#
# This script is used for collecting all itemids in the database. Itemids are ids of features.
#
# In this task we only collect itemids from the following tables:
# - inputevents
# - outputevents
# - chartevents
# - labevents
# - microbiologyevents
# - prescriptions

# %%
from __future__ import print_function

import psycopg2
import datetime
import sys
from operator import itemgetter, attrgetter, methodcaller
import numpy as np
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool, Process, Lock
from multiprocessing.sharedctypes import Value, Array
from ctypes import Structure, c_double, c_int
from tqdm import tqdm as tqdm

from preprocessing.utils import getConnection

# function to execute a sql and store result to specific location of an array, used for parallel query


def _queryAndStoreSql(sqls_itemids):
    tconn = getConnection()
    subresults = []
    for sql, itemid in tqdm(sqls_itemids):
        tcur = tconn.cursor()
        tcur.execute(sql)
        r2 = tcur.fetchall()
        subresults.append((itemid, int(r2[0][0])))
    tconn.close()
    return subresults


def _getNumberOfAdmissionThatUseStatId(sql, itemids, admission_ids_txt, savefile, numworkers):
    starttime = datetime.datetime.now()
    p = Pool(numworkers)
#     numberOfAdmissionThatUseItemid = [(0, 0) for t in range(len(itemids))]
    numberOfAdmissionThatUseItemid = []
    sqls_itemids = []
    for t, itemid in enumerate(itemids):
        itemidstr = '= {0}'.format(itemid)
        if itemid is None:
            itemidstr = 'is null'
        sqls_itemids.append((sql.format(itemidstr), itemid))
    sqls_itemids_units = np.array_split(sqls_itemids, numworkers)
    for sqls_itemids_unit in sqls_itemids_units:
        numberOfAdmissionThatUseItemid.append(p.apply_async(_queryAndStoreSql,
                                                            args=(sqls_itemids_unit,)))
    p.close()
    p.join()

    numberOfAdmissionThatUseItemid = [
        t.get() for t in numberOfAdmissionThatUseItemid]
    numberOfAdmissionThatUseItemid = list(
        itertools.chain.from_iterable(numberOfAdmissionThatUseItemid))
    numberOfAdmissionThatUseItemid = sorted(
        numberOfAdmissionThatUseItemid, key=lambda tup: tup[1])
    numberOfAdmissionThatUseItemid.reverse()
    np.save(savefile, numberOfAdmissionThatUseItemid)


def _getNumberOfAdmissionThatUseStatIdBio(itemids, admission_ids_txt, savefile, numworkers):
    starttime = datetime.datetime.now()
    p = Pool(numworkers)
    numberOfAdmissionThatUseItemid = []
    sqls_itemids = []
    for t, itemid in enumerate(itemids):
        sql = 'select count(distinct hadm_id) from mimiciii.microbiologyevents where hadm_id in (select * from admission_ids)'
        if (itemid[0] != None):
            sql += ' and spec_itemid=' + str(itemid[0])
        else:
            sql += ' and spec_itemid is null'
        if (itemid[1] != None):
            sql += ' and org_itemid=' + str(itemid[1])
        else:
            sql += ' and org_itemid is null'
        if (itemid[2] != None):
            sql += ' and ab_itemid=' + str(itemid[2])
        else:
            sql += ' and ab_itemid is null'
        sqls_itemids.append((sql, itemid))
    sqls_itemids_units = np.array_split(sqls_itemids, numworkers)
    for sqls_itemids_unit in sqls_itemids_units:
        numberOfAdmissionThatUseItemid.append(p.apply_async(_queryAndStoreSql,
                                                            args=(sqls_itemids_unit,)))
    p.close()
    p.join()

    numberOfAdmissionThatUseItemid = [
        t.get() for t in numberOfAdmissionThatUseItemid]
    numberOfAdmissionThatUseItemid = list(
        itertools.chain.from_iterable(numberOfAdmissionThatUseItemid))
    numberOfAdmissionThatUseItemid = sorted(
        numberOfAdmissionThatUseItemid, key=lambda tup: tup[1])
    numberOfAdmissionThatUseItemid.reverse()
    np.save(savefile, numberOfAdmissionThatUseItemid)


def _getNumberOfAdmissionThatUseStatIdPrescript(sql, itemids, admission_ids_txt, savefile, numworkers):
    starttime = datetime.datetime.now()
    p = Pool(numworkers)
    numberOfAdmissionThatUseItemid = []
    sqls_itemids = []
    for t, itemid in enumerate(itemids):
        itemidstr = '= \'{0}\''.format(itemid)
        if itemid is None:
            itemidstr = 'is null'
        sqls_itemids.append((sql.format(itemidstr), itemid))
    sqls_itemids_units = np.array_split(sqls_itemids, numworkers)
    for sqls_itemids_unit in sqls_itemids_units:
        numberOfAdmissionThatUseItemid.append(p.apply_async(_queryAndStoreSql,
                                                            args=(sqls_itemids_unit,)))
    p.close()
    p.join()

    numberOfAdmissionThatUseItemid = [
        t.get() for t in numberOfAdmissionThatUseItemid]
    numberOfAdmissionThatUseItemid = list(
        itertools.chain.from_iterable(numberOfAdmissionThatUseItemid))
    numberOfAdmissionThatUseItemid = sorted(
        numberOfAdmissionThatUseItemid, key=lambda tup: tup[1])
    numberOfAdmissionThatUseItemid.reverse()
    np.save(savefile, numberOfAdmissionThatUseItemid)


def getItemIdList(args):
    print('1_getItemIdList: Select all itemids from TABLE INPUTEVENTS, OUTPUTEVENTS, CHARTEVENTS, LABEVENTS, MICROBIOLOGYEVENTS, PRESCRIPTIONS. ')

    # %%
    conn = getConnection()

    # %%
    # load admission_ids
    cachedir = Path(args.cachedir)
    _adm = np.load(cachedir.joinpath('res/admission_ids.npy'),
                   allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    # ## Itemids from inputevents
    #
    # Data from Carevue and Metavision is separately stored in TABLE INPUTEVENTS_CV and TABLE INPUTEVENTS_MV. Inputevents from Metavision have itemids >= 200000, and those from Carevue have itemids in [30000, 49999].

    # %%
    # itemid from inputevents
    # sql = 'select distinct itemid from mimiciii.inputevents_cv where itemid >= 30000 and itemid <= 49999'
    sql = '''
    with inputitemids as (
            select distinct itemid from mimiciii.inputevents_mv where itemid >= 200000
            union
            select distinct itemid from mimiciii.inputevents_cv where itemid >= 30000 and itemid <= 49999
        )
    select distinct itemid from inputitemids
    '''
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    input_itemid = [r[0] for r in res]
    input_itemid_txt = ','.join(map(str, input_itemid))

    print("len(input_itemid) = ", len(input_itemid))

    # ## Itemids from outputevents
    #
    # We only need to collect all distinct itemids in TABLE OUTPUTEVENTS.

    # %%
    # itemid from outputevents
    # sql = 'select distinct itemid from mimiciii.outputevents where itemid >= 30000 and itemid <= 49999'
    sql = 'select distinct itemid from mimiciii.outputevents'
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    output_itemid = [r[0] for r in res]
    output_itemid_txt = ','.join(map(str, output_itemid))

    print("len(output_itemid) = ", len(output_itemid))

    # ## Itemids from chartevents
    #
    # We only need to collect all distinct itemids in TABLE CHARTEVENTS.

    # %%
    # itemid from chartevents, should collect all ids <= 49999
    # sql = 'select distinct itemid from mimiciii.chartevents where itemid <= 49999'
    sql = 'select distinct itemid from mimiciii.chartevents'
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    chart_itemid = [r[0] for r in res]
    chart_itemid_txt = ','.join(map(str, chart_itemid))

    print("len(chart_itemid) = ", len(chart_itemid))

    # ## Itemids from labevents
    #
    # We only need to collect all distinct itemids in TABLE LABEVENTS.

    # %%
    # itemid from labevenets
    sql = 'select distinct itemid from mimiciii.labevents'
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    lab_itemid = [r[0] for r in res]
    lab_itemid_txt = ','.join(map(str, lab_itemid))

    print("len(lab_itemid) = ", len(lab_itemid))

    # ## Itemids from microbiologyevents
    #
    # We need to collect 4 kinds of itemids:
    # - spec_itemid
    # - org_itemid
    # - ab_itemid
    # - tuple of all above

    # %%
    # itemid from microbiologyevents
    sql = 'select distinct (spec_itemid,org_itemid,ab_itemid),spec_itemid,org_itemid,ab_itemid from mimiciii.microbiologyevents'
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    microbio_itemid = []
    for r in res:
        ele = r[0][1:-1].split(',')
        for t in range(len(ele)):
            try:
                ele[t] = int(ele[t])
            except:
                ele[t] = None
        microbio_itemid.append(tuple(ele))

    print("len(microbio_itemid) = ", len(microbio_itemid))

    # ## Itemids from prescriptions
    #
    # We only need to collect all distinct itemids in TABLE PRESCRIPTIONS.

    # %%
    # itemid from prescriptions
    sql = 'select distinct formulary_drug_cd from mimiciii.prescriptions'
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    prescript_itemid = [r[0] for r in res]

    print("len(prescript_itemid) = ", len(prescript_itemid))

    # %%
    database = {'input': input_itemid,
                'output': output_itemid,
                'chart': chart_itemid,
                'lab': lab_itemid,
                'microbio': microbio_itemid,
                'prescript': prescript_itemid}
    np.save(cachedir.joinpath('res/itemids.npy'), database)
    print('saved!')

    # ## Histograms of itemids
    #
    # For each table we draw the histogram showing the number of admissions which have any record of each itemid.

    # %%

    # load itemids
    itemids = np.load(cachedir.joinpath('res/itemids.npy'),
                      allow_pickle=True).tolist()

    # labevent histogram
    # print(itemids['lab'])
    sql = 'select count(distinct hadm_id) from mimiciii.labevents where itemid {0} AND hadm_id in (select * from admission_ids)'
    _getNumberOfAdmissionThatUseStatId(sql, itemids['lab'], admission_ids_txt,
                                      cachedir.joinpath('res/labevent_numberOfAdmissionThatUseItemid.npy'), args.num_workers)

    # microbio histogram
    # print(itemids['microbio'])
    _getNumberOfAdmissionThatUseStatIdBio(itemids['microbio'], admission_ids_txt,
                                         cachedir.joinpath('res/microbio_numberOfAdmissionThatUseItemid.npy'), args.num_workers)

    # prescript histogram
    # print(itemids['prescript'])
    sql = 'select count(distinct hadm_id) from mimiciii.prescriptions where formulary_drug_cd {0} and hadm_id in (select * from admission_ids)'
    _getNumberOfAdmissionThatUseStatIdPrescript(sql, itemids['prescript'], admission_ids_txt,
                                               cachedir.joinpath('res/prescript_numberOfAdmissionThatUseItemid.npy'), args.num_workers)

    # %%
    # finish stats in seperate py file for the convenience of multi-processing
    figpath = cachedir.joinpath('figure')
    figpath.mkdir(parents=True, exist_ok=True)
    labevent_histo = np.load(cachedir.joinpath(
        'res/labevent_numberOfAdmissionThatUseItemid.npy'), allow_pickle=True).tolist()
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(labevent_histo))], [int(r[1])
                                                      for r in labevent_histo])
    plt.title('Number of Admission That Use Itemid: labevent')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')
    plt.savefig(figpath.joinpath(
        'labevent_numberOfAdmissionThatUseItemid.pdf'))

    # %%
    microbio_histo = np.load(cachedir.joinpath(
        'res/microbio_numberOfAdmissionThatUseItemid.npy'), allow_pickle=True).tolist()
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(microbio_histo))], [int(r[1])
                                                      for r in microbio_histo])
    plt.title('Number of Admission That Use Itemid: microbioevent')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')

    # %%
    microbio_histo[:200]
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(microbio_histo[:200]))], [int(r[1])
                                                            for r in microbio_histo[:200]])
    plt.title('Number of Admission That Use Itemid: microbioevent(top 200)')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')
    plt.savefig(figpath.joinpath(
        'microbio_numberOfAdmissionThatUseItemid.pdf'))

    # %%
    prescript_histo = np.load(cachedir.joinpath(
        'res/prescript_numberOfAdmissionThatUseItemid.npy'), allow_pickle=True).tolist()
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(prescript_histo))], [int(r[1])
                                                       for r in prescript_histo])
    plt.title('Number of Admission That Use Itemid: prescriptionevent')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')
    plt.savefig(figpath.joinpath(
        'prescript_numberOfAdmissionThatUseItemid.pdf'))

    # %%
    # inputevent histogram
    sql = 'select sum(count) from (select count(distinct hadm_id) as count from mimiciii.inputevents_mv where itemid {0} and hadm_id in (select * from admission_ids) union all select count(distinct hadm_id) as count from mimiciii.inputevents_cv where itemid {0} and hadm_id in (select * from admission_ids) ) t'
    _getNumberOfAdmissionThatUseStatId(sql, itemids['input'], admission_ids_txt,
                                      cachedir.joinpath('res/inputevent_numberOfAdmissionThatUseItemid.npy'), args.num_workers)

    print('input finished')

    # outputevent histogram
    sql = 'select count(distinct hadm_id) from mimiciii.outputevents where itemid {0} AND hadm_id in (select * from admission_ids)'
    _getNumberOfAdmissionThatUseStatId(sql, itemids['output'], admission_ids_txt,
                                      cachedir.joinpath('res/outputevent_numberOfAdmissionThatUseItemid.npy'), args.num_workers)

    print('output finished')

    # chartevent
    sql = 'select count(distinct hadm_id) from mimiciii.chartevents where itemid {0} AND hadm_id in (select * from admission_ids)'
    _getNumberOfAdmissionThatUseStatId(sql, itemids['chart'], admission_ids_txt,
                                      cachedir.joinpath('res/chartevent_numberOfAdmissionThatUseItemid.npy'), args.num_workers)

    print('chart finished')

    # %%
    histo = np.load(cachedir.joinpath(
        'res/inputevent_numberOfAdmissionThatUseItemid.npy'), allow_pickle=True).tolist()
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(histo))], [int(r[1]) for r in histo])
    plt.title('Number of Admission That Use Itemid: inputevent')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')
    plt.savefig(figpath.joinpath(
        'inputevent_numberOfAdmissionThatUseItemid.pdf'))

    # %%
    histo = np.load(cachedir.joinpath(
        'res/outputevent_numberOfAdmissionThatUseItemid.npy'), allow_pickle=True).tolist()
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(histo))], [int(r[1]) for r in histo])
    plt.title('Number of Admission That Use Itemid: outputevent')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')
    plt.savefig(figpath.joinpath(
        'outputevent_numberOfAdmissionThatUseItemid.pdf'))

    # %%
    histo = np.load(cachedir.joinpath(
        'res/chartevent_numberOfAdmissionThatUseItemid.npy'), allow_pickle=True).tolist()
    plt.figure(figsize=(10, 5))
    plt.bar([i for i in range(len(histo))], [int(r[1]) for r in histo])
    plt.title('Number of Admission That Use Itemid: chartevent')
    plt.xlabel(
        'the rank of feature, ordered by number of admissions using this feature desc')
    plt.ylabel('number of admissions using this feature')
    plt.savefig(figpath.joinpath(
        'chartevent_numberOfAdmissionThatUseItemid.pdf'))

    print('Finished 1_getItemIdList!')
    print()
