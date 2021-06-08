# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# # Get 136 raw features
#
# In this script we extracts 136 raw features from datafiles we generated in 8_processing.ipynb.
#
# After this step, we have the following output files:
# - DB_merged_Xhrs.npy: matrices of time series of each admission. For each admission, there is a matrix containing its records in given time period. Each row of the matrix is like this: [feature 0, …, feature n, number of seconds in [icu intime, current time ], admission_id].
# - ICD9-Xhrs.npy: matrices of ICD9 codes of each admission. For each admission, there is a matrix containing its ICD9 codes. Each line of the matrix is like this: [admission_id, icd9 original code, icd9 parsed code, icd9 subcat number]
# - AGE_LOS_MORTALITY_Xhrs.npy: matrices of the result of AGE_LOS_MORTALITY function for each admission. Here we just keep it for compatibility.
# - ADM_FEATURES_Xhrs.npy: features only related to admissions and not related to time, containing age, whether there is AIDS/hematologic malignancy/metastatic cancer and admission type.
# - ADM_LABELS_Xhrs.npy: mortality labels of all admissions, containing hospital mortality, 1/2/3-day mortality, 30-day mortality and 1-year mortality.

# %%
from __future__ import print_function
import csv

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
from collections import OrderedDict
from pathlib import Path

from preprocessing.utils import getConnection
from preprocessing.utils import parseUnitsMap
from preprocessing.utils import parseNum
from preprocessing.utils import sparsify


# %%

def extract_time_series(aidres):
    time_series = aidres['serial_features']
    tn = time_series['timestep']
    fn = time_series['features']
    codes = time_series['codes']
    series = [[None for ft in range(fn)] for tt in range(tn)]
    for tt, ft, value in codes:
        try:
            series[tt][ft] = value
        except:
            print(tt, ft, value, tn, fn)
    return series


def check_adm_hrs_pass(admres, hrs):
    return admres['serial_features']['timelength'] > hrs * 3600.0


def extract_data(aid, hrs, RAWDIR, SOURCEDIR):
    admres = np.load(os.path.join(RAWDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
    if check_adm_hrs_pass(admres, hrs):
        ori_admres = np.load(os.path.join(
            SOURCEDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
        return (
            extract_time_series(admres),
            ori_admres['icd9'],
            ori_admres['general'],
            admres['adm_features'],
            admres['adm_labels']
        )
    else:
        return None


def mapping_features(fids, mapped, map_itemid_index):
    for key, value in fids.items():
        temp = mapping_lists(value, map_itemid_index)
        if temp is not None:
            mapped[key] = temp


def mapping_lists(fidlist, map_itemid_index):
    if type(fidlist) == type([]):
        mapped = []
        for fid in fidlist:
            temp = mapping_lists(fid, map_itemid_index)
            if temp is not None:
                mapped.append(temp)
        if len(mapped) > 0:
            return mapped
        else:
            return None
    elif type(fidlist) == type(()):
        temp = mapping_lists(fidlist[1], map_itemid_index)
        if temp is not None:
            return (fidlist[0], temp)
        else:
            return None
    else:
        assert type(fidlist) == type('str') or type(fidlist) == type(0)
        try:
            res = map_itemid_index[fidlist][0]
            return res
        except:
            return None


def keep_nonneg(x):
    try:
        x = float(x)
        return x >= 0
    except:
        return True


def merge_items(timeseries, mergenode, new_column_id, keep_func=keep_nonneg):
    # record timestamps
    if type(mergenode) == type(()):
        merged = {}
        mergelist = mergenode[1]
        mergename = mergenode[0]

        if mergename.endswith('_div'):
            def merge_func(x): return x[0] / x[1]
        elif mergename.endswith('_min'):
            merge_func = np.nanmin
        elif mergename.endswith('_max'):
            merge_func = np.nanmax
        elif mergename.endswith('_sum'):
            merge_func = np.nansum
        elif mergename.endswith('_mean'):
            merge_func = np.nanmean
        elif mergename.endswith('_f2c'):
            def merge_func(x): return (np.nanmean(x) - 32) * 5.0 / 9.0
        elif mergename.endswith('_lb2kg'):
            def merge_func(x): return np.nanmean(x) * 0.453592
        elif mergename.endswith('_oz2kg'):
            def merge_func(x): return np.nanmean(x) * 0.0283495
        elif mergename.endswith('_inches2cm'):
            def merge_func(x): return np.nanmean(x) * 2.54
        else:
            merge_func = np.nanmean

        for mergeitem in mergelist:
            subtree = merge_items(timeseries, mergeitem,
                                  new_column_id, keep_func=keep_func)
            for record in subtree:
                try:
                    merged[record[0]].append(record[2])
                except KeyError:
                    merged[record[0]] = [record[2]]
        for key in merged.keys():
            try:
                merged[key] = merge_func(merged[key])
            except:
                merged[key] = None
        return [(key, new_column_id, value) for key, value in merged.items()]
    else:
        assert type(mergenode) == type('string') or type(mergenode) == type(0)
        subtree = []
        for record in timeseries:
            i, j, value = record[0], record[1], record[2]
            if j == mergenode:
                if value is not None and keep_func(value):
                    subtree.append((i, new_column_id, value))
        return subtree


def extract_serial_features(aid, res, mapped_feature_itemids, map_feature_colids):
    timeseries, general, icd9 = res['timeseries']['codes'], res['general'], res['icd9']
    D = res['timeseries']['features']
    assert general[0] == aid
    new_timeseries = []
    for featurename, featurelist in mapped_feature_itemids.items():
        temp = merge_items(timeseries, (featurename, featurelist),
                           map_feature_colids[featurename])
        if temp is not None and len(temp) > 0:
            new_timeseries.append(temp)

    # get timestamps
    timecolid = len(map_feature_colids)
    valid_timestamps = list(itertools.chain(
        *[[t[0] for t in tt] for tt in new_timeseries]))
    max_timestamp = max(valid_timestamps)
    timestamps = [(t[0], timecolid, t[2])
                  for t in timeseries if t[0] <= max_timestamp and t[1] == D - 2]
    new_timeseries.append(timestamps)
    timelength = max([t[2] for t in timestamps]) - \
        min([t[2] for t in timestamps])

    # aid
    aidcolid = timecolid + 1
    aids = [(t[0], aidcolid, t[2])
            for t in timeseries if t[0] <= max_timestamp and t[1] == D - 1]
    new_timeseries.append(aids)

    res2 = list(itertools.chain(*new_timeseries))
    return {'timestep': max_timestamp + 1, 'features': aidcolid+1, 'codes': res2, 'timelength': timelength}


def extract_adm_features(processed_adm):
    # get from previous in database
    return processed_adm['adm_features']


def extract_adm_labels(aid, LABELDIR):
    admlabel = np.load(os.path.join(LABELDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
    adm_labels = (
        admlabel['mor'],
        admlabel['mor24'],
        admlabel['mor48'],
        admlabel['mor72'],
        admlabel['mor30d'],
        admlabel['mor1y'],
    )
    return adm_labels


def extract_adm(aid, SOURCEDIR, PROCESSED_DB_DIR, RAWDIR, mapped_feature_itemids, map_feature_colids, LABELDIR):
    admres = np.load(os.path.join(
        SOURCEDIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), allow_pickle=True).tolist()
    processed_admres = np.load(os.path.join(
        PROCESSED_DB_DIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), allow_pickle=True).tolist()
    res = {
        'serial_features': extract_serial_features(aid, admres, mapped_feature_itemids, map_feature_colids),
        'adm_features': extract_adm_features(processed_admres),
        'adm_labels': extract_adm_labels(aid, LABELDIR)
    }
    np.save(os.path.join(RAWDIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), res)
    print('finished {0}!'.format(aid))


def get_99plus_features_raw(args):
    cachedir = Path(args.cachedir)
    RESDIR = cachedir.joinpath('admdata_99p')
    SOURCEDIR = cachedir.joinpath('admdata_valid')
    LABELDIR = cachedir.joinpath('admdata_timelabels')
    PROCESSED_DB_DIR = cachedir.joinpath('admdata_17f/processed_db')
    RAWDIR = os.path.join(RESDIR, 'raw')
    for dname in [RESDIR, RAWDIR]:
        if not os.path.exists(dname):
            os.makedirs(dname)

    # %%
    # load the aid used for previous exps
    tempdball = np.load(os.path.join(
        RESDIR, '../admdata_17f/24hrs/DB_merged_24hrs.npy'), allow_pickle=True)
    valid_aids = [x[0][-1] for x in tempdball]
    print(len(valid_aids))
    print(valid_aids[:10])

    # %%
    map_itemid_index = np.load(cachedir.joinpath('res/map_itemid_index.npy'), allow_pickle=True).tolist()

    # ## Define the map between features and itemids
    #
    # Since the number of features is too big, we use an AST(abstract syntax tree) structure to define the mapping between features and itemids and the manipulation on features.
    #
    # We use a dictionary as the top sturcture, the key is the name of the feature and the value is marked as L.
    #
    # We define L as follows:
    #
    # ```
    # L -> list of [ID, MANI]
    # ID -> a specific itemid
    # MANI -> tuple(name of manipulation, L)
    # ```
    #
    # With this definition, we can easily parse the map recursively.
    #
    # We conduct different manipulation according to the end of name of manipulation, and we define following methods:
    # - min
    # - max
    # - sum
    # - mean
    # - f2c: convert farenheit to celcius and use mean when in confliction
    # - lb2kg: convert lb to kg and use mean when in confliction
    # - oz2kg: convert oz to kg and use mean when in confliction
    # - inches2cm: convert inch to cm and use mean when in confliction
    #
    # Other methods can be added. However, all methods should accept a list and return a single value.

    # %%
    # readin the csv file recording itemids
    feature_itemids = OrderedDict({})

    # added features
    with open('preprocessing/config/99plusf.csv', 'r') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='\"')
        next(csvreader)
        for row in csvreader:
            feature_name = ''.join(row[3:])
            itemids = row[2].split(',')[:-1]
            inmapitemids = []
            for itemid in itemids:
                try:
                    itemid = int(itemid)
                except:
                    pass
                if itemid in map_itemid_index.keys():
                    inmapitemids.append(itemid)
            if len(inmapitemids) > 0:
                feature_itemids[feature_name] = inmapitemids

    # features from previous 17 raw features
    feature_itemids_17raw = OrderedDict([
        ['gcsverbal', [723, 223900]],
        ['gcsmotor', [454, 223901]],
        ['gcseyes', [184, 220739]],
        ['systolic_blood_pressure_abp_mean', [51, 442, 455, 6701, 220050, 220179]],
        ['heart_rate', [211, 220045]],
        ['body_temperature', [
            ('_f2c', [678, 223761]),
            676,
            223762
        ]],
        ['pao2', [50821]],
        ['fio2', [50816, 223835, 3420, 3422, 190]],
        ['urinary_output_sum', [40055,
                                43175,
                                40069,
                                40094,
                                40715,
                                40473,
                                40085,
                                40057,
                                40056,
                                40405,
                                40428,
                                40086,
                                40096,
                                40651,
                                226559,
                                226560,
                                226561,
                                226584,
                                226563,
                                226564,
                                226565,
                                226567,
                                226557,
                                226558,
                                227488,
                                227489]],
        ['serum_urea_nitrogen_level', [51006]],
        ['white_blood_cells_count_mean', [51300, 51301]],
        ['serum_bicarbonate_level_mean', [50882]],
        ['sodium_level_mean', [50824, 50983]],
        ['potassium_level_mean', [50822, 50971]],
        ['bilirubin_level', [50885]],
    ])

    for key, value in feature_itemids_17raw.items():
        feature_itemids[key] = value

    # add the ratio of inspiratory:expiratory
    feature_itemids['ie_ratio_mean'] = [
        221,
        ('_div', [226873, 226871])
    ]

    # add other features
    extra_features = OrderedDict([
        ['diastolic_blood_pressure_mean', [8368, 8440, 8441, 8555, 220180, 220051]],
        ['arterial_pressure_mean', [456, 52, 6702, 443, 220052, 220181, 225312]],
        ['respiratory_rate', [618, 615, 220210, 224690]],
        ['spo2_peripheral', [646, 220277]],
        ['glucose', [807, 811, 1529, 3745, 3744, 225664, 220621, 226537]],
        ['weight', [
            762, 763, 3723, 3580, 226512,
            ('_lb2kg', [3581]),
            ('_oz2kg', [3582])
        ]],
        ['height', [
            ('_inches2cm', [920, 1394, 4187, 3486, ]),
            3485, 4188, 226707
        ]],
        ['hgb', [50811, 51222]],
        ['platelet', [51265]],
        ['chloride', [50806, 50902]],
        ['creatinine', [50912]],
        ['norepinephrine', [30047, 30120, 221906]],
        ['epinephrine', [30044, 30119, 30309, 221289]],
        ['phenylephrine', [30127, 30128, 221749]],
        ['vasopressin', [30051, 222315]],
        ['dopamine', [30043, 30307, 221662]],
        ['isuprel', [30046, 227692]],
        ['midazolam', [30124, 221668]],
        ['fentanyl', [30150, 30308, 30118, 30149, 221744, 225972, 225942]],
        ['propofol', [30131, 222168]],
        ['peep', [50819]],
        ['ph', [50820]],
    ])

    for key, value in extra_features.items():
        feature_itemids[key] = value

    print(feature_itemids)

    # %%
    # from copy import deepcopy
    # mapped_feature_itemids = deepcopy(feature_itemids)
    mapped_feature_itemids = OrderedDict({})

    mapping_features(feature_itemids, mapped_feature_itemids, map_itemid_index)
    print(mapped_feature_itemids)

    map_feature_colids = {}
    t = 0
    for key in mapped_feature_itemids.keys():
        map_feature_colids[key] = t
        t += 1
    print(len(map_feature_colids))
    print(map_feature_colids)

    # %%
    np.save(cachedir.joinpath('admdata_99p/raw/',
                         'map_feature_colids.npy'), map_feature_colids)

    # %%
    # merge selected items to one item; the value is the mean of all values
    # Here we parse the feature_itemids like a grammar tree with top-down method, thus we can directly define our manipulation in the dict
    # dict stands for non-leaf nodes and non-dict stands for leaf nodes
    # map itemids to column numbers
    # def get_column_set(itemids, map_itemid_index):
    #     return set([map_itemid_index[itemid][0] for itemid in itemids])

    demoseries = [
        [0, 0, 1],
        [0, 1, 3],
        [0, 2, 4]
    ]
    demomerge = {
        'ratio': [
            0,
            ('pf_div', [1, 2])
        ]
    }

    for key, value in demomerge.items():
        print(merge_items(demoseries, (key, value), 2233))

    # %%

    # %%

    # %%
    p = Pool(args.num_workers)
    for aid in valid_aids:
        p.apply_async(extract_adm, args=(aid, SOURCEDIR, PROCESSED_DB_DIR,
                                         RAWDIR, mapped_feature_itemids, map_feature_colids, LABELDIR))
    p.close()
    p.join()

    # ## Select admissions with > xxhrs records
    #
    # In this step we only keep admissions with record length > 24/48 hrs.

    # %%
    TARGETDIR = RESDIR

    def collect_admissions_with_more_than_hrs(hrs):
        processed_valid_aids = valid_aids

        HRDIR = os.path.join(TARGETDIR, '%dhrs_raw' % hrs)
        if not os.path.exists(HRDIR):
            os.makedirs(HRDIR)

        p = Pool(args.num_workers)
        collec = [p.apply_async(extract_data, args=(aid, hrs, RAWDIR, SOURCEDIR))
                  for aid in processed_valid_aids]
        p.close()
        p.join()
        collec = [x.get() for x in collec]
        collec = [x for x in collec if x is not None]

        data_all = [r[0] for r in collec]
        label_icd9_all = [r[1] for r in collec]
    #     label_mor_all = [r[2][:6] for r in collec]
        label_mor_all = [r[2] for r in collec]
        adm_features_all = [r[3] for r in collec]
        adm_labels_all = [r[4] for r in collec]

        np.save(os.path.join(HRDIR, 'DB_merged_%dhrs.npy' % hrs), data_all)
        np.save(os.path.join(HRDIR, 'ICD9-%dhrs.npy' % hrs), label_icd9_all)
        np.save(os.path.join(HRDIR, 'AGE_LOS_MORTALITY_%dhrs.npy' %
                             hrs), label_mor_all)
        np.save(os.path.join(HRDIR, 'ADM_FEATURES_%dhrs.npy' %
                             hrs), adm_features_all)
        np.save(os.path.join(HRDIR, 'ADM_LABELS_%dhrs.npy' %
                             hrs), adm_labels_all)

    # >= 24hrs
    collect_admissions_with_more_than_hrs(24)
    collect_admissions_with_more_than_hrs(48)
