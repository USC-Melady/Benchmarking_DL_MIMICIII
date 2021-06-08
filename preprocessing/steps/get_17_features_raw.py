# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# # Get 17 raw features
#
# In this script we extracts 17 raw features from datafiles we generated in 8_processing.ipynb.
#
# After this step, we have the following output files:
# - DB_merged_Xhrs.npy: matrices of time series of each admission. For each admission, there is a matrix containing its records in given time period. Each row of the matrix is like this: [feature 0, …, feature n, number of seconds in [icu intime, current time ], admission_id].
# - ICD9-Xhrs.npy: matrices of ICD9 codes of each admission. For each admission, there is a matrix containing its ICD9 codes. Each line of the matrix is like this: [admission_id, icd9 original code, icd9 parsed code, icd9 subcat number]
# - AGE_LOS_MORTALITY_Xhrs.npy: matrices of the result of AGE_LOS_MORTALITY function for each admission. Here we just keep it for compatibility.
# - ADM_FEATURES_Xhrs.npy: features only related to admissions and not related to time, containing age, whether there is AIDS/hematologic malignancy/metastatic cancer and admission type.
# - ADM_LABELS_Xhrs.npy: mortality labels of all admissions, containing hospital mortality, 1/2/3-day mortality, 30-day mortality and 1-year mortality.

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
import traceback
import shutil
from collections import OrderedDict
from pathlib import Path

from preprocessing.utils import getConnection
from preprocessing.utils import parseUnitsMap
from preprocessing.utils import parseNum
from preprocessing.utils import sparsify


def keep_nonneg(x):
    try:
        x = float(x)
        return x >= 0
    except:
        return True


def merge_items(timeseries, mergeset, new_column_id, merge_func=np.mean, keep_func=keep_nonneg):
    merged = {}
    for record in timeseries:
        i, j, value = record[0], record[1], record[2]
        if j in mergeset:
            #             try:
            #                 value = value
            #             except:
            #                 pass
            try:
                if value is not None and keep_func(value):
                    merged[i].append(value)
            except KeyError:
                merged[i] = [value]
    for key in merged.keys():
        try:
            merged[key] = merge_func(merged[key])
        except:
            merged[key] = merged[key][0]
    return [(key, new_column_id, value) for key, value in merged.items()]

# map itemids to column numbers


def get_column_set(itemids, map_itemid_index):
    return set([map_itemid_index[itemid][0] for itemid in itemids])


def extract_serial_features(aid, res, feature_itemids, map_itemid_index, map_feature_colids, merge_funcs):
    timeseries, general, icd9 = res['timeseries']['codes'], res['general'], res['icd9']
    D = res['timeseries']['features']
    assert general[0] == aid
    new_timeseries = []

    # body temperature
    ctemp = merge_items(
        timeseries,
        get_column_set(
            feature_itemids['body_temperature']['c'], map_itemid_index),
        map_feature_colids['body_temperature']
    )
    ctempids = set([x[0] for x in ctemp])

    ftemp = merge_items(
        timeseries,
        get_column_set(
            feature_itemids['body_temperature']['f'], map_itemid_index),
        map_feature_colids['body_temperature']
    )
    ftemp = [(x[0], x[1], (x[2] - 32) * 5 / 9.0)
             for x in ftemp if x[0] not in ctempids]
    ftemp = [x for x in ftemp if x[2] > 0]
    new_timeseries.append(ctemp + ftemp)

    # urinary output
    new_timeseries.append(merge_items(
        timeseries,
        get_column_set(feature_itemids['urinary_output'], map_itemid_index),
        map_feature_colids['urinary_output'],
        merge_func=np.sum
    ))

    # serum urea nitrogen/white blood cells count/serum bicarbonate/sodium/potassium/bilirubin level, just merge
    for itemname in ['gcsverbal',
                     'gcsmotor',
                     'gcseyes',
                     'systolic_blood_pressure_abp_mean',
                     'heart_rate',
                     'pao2',
                     'fio2',
                     'serum_urea_nitrogen_level',
                     'white_blood_cells_count',
                     'serum_bicarbonate_level',
                     'sodium_level',
                     'potassium_level',
                     'bilirubin_level']:
        tks = [tk for tk in feature_itemids.keys() if tk.startswith(itemname)]
        for tk in tks:
            try:
                merge_func = merge_funcs[tk.split('_')[-1]]
            except:
                merge_func = np.mean
            new_timeseries.append(merge_items(
                timeseries,
                get_column_set(feature_itemids[tk], map_itemid_index),
                map_feature_colids[tk],
                merge_func=merge_func
            ))

    # get timestamps
    valid_timestamps = list(itertools.chain(
        *[[t[0] for t in tt] for tt in new_timeseries]))
    max_timestamp = max(valid_timestamps)
    timestamps = [(t[0], map_feature_colids['timestamp'], t[2])
                  for t in timeseries if t[0] <= max_timestamp and t[1] == D - 2]
    new_timeseries.append(timestamps)
    timelength = max([t[2] for t in timestamps]) - \
        min([t[2] for t in timestamps])

    # aid
    aids = [(t[0], map_feature_colids['aid'], t[2])
            for t in timeseries if t[0] <= max_timestamp and t[1] == D - 1]
    new_timeseries.append(aids)

    res2 = list(itertools.chain(*new_timeseries))
    return {'timestep': max_timestamp + 1, 'features': len(map_feature_colids), 'codes': res2, 'timelength': timelength}


def extract_adm_features(processed_adm):
    # get from previous in database
    return processed_adm['adm_features']


def extract_adm_labels(aid, LABELDIR):
    admlabel = np.load(os.path.join(LABELDIR, 'adm-%.6d.npy' %
                                    aid), allow_pickle=True).tolist()
    adm_labels = (
        admlabel['mor'],
        admlabel['mor24'],
        admlabel['mor48'],
        admlabel['mor72'],
        admlabel['mor30d'],
        admlabel['mor1y'],
    )
    return adm_labels


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
    admres = np.load(os.path.join(RAWDIR, 'adm-%.6d.npy' %
                                  aid), allow_pickle=True).tolist()
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


def extract_adm(aid, SOURCEDIR, PROCESSED_DB_DIR, RAWDIR, feature_itemids, map_itemid_index, map_feature_colids, merge_funcs, LABELDIR):
    admres = np.load(os.path.join(
        SOURCEDIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), allow_pickle=True).tolist()
    processed_admres = np.load(os.path.join(
        PROCESSED_DB_DIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), allow_pickle=True).tolist()
    res = {
        'serial_features': extract_serial_features(aid, admres, feature_itemids, map_itemid_index, map_feature_colids, merge_funcs),
        'adm_features': extract_adm_features(processed_admres),
        'adm_labels': extract_adm_labels(aid, LABELDIR)
    }
    np.save(os.path.join(
        RAWDIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), res)
    print('finished {0}!'.format(aid))


# # First to extract 17 features

# %%
def get_17_features_raw(args):
    cachedir = Path(args.cachedir)
    SOURCEDIR = cachedir.joinpath('admdata_valid')
    TARGETDIR = cachedir.joinpath('admdata_17f')
    LABELDIR = cachedir.joinpath('admdata_timelabels')
    RAWDIR = os.path.join(TARGETDIR, 'raw')
    PROCESSED_DB_DIR = os.path.join(TARGETDIR, 'processed_db')

    if not os.path.exists(TARGETDIR):
        os.makedirs(TARGETDIR)
    if not os.path.exists(RAWDIR):
        os.makedirs(RAWDIR)

    valid_aids = [re.match(r'adm\-(\d+)\.npy', x)
                  for x in os.listdir(SOURCEDIR)]
    valid_aids = sorted([int(x.group(1)) for x in valid_aids if x is not None])
    print(len(valid_aids), valid_aids[:10])

    map_itemid_index = np.load(cachedir.joinpath(
        'res/map_itemid_index.npy'), allow_pickle=True).tolist()

    # %%
    # merge selected items to one item; the value is the mean of all values

    # test on Glasgow coma scale
    adm = np.load(os.path.join(SOURCEDIR, 'adm-194627.npy'),
                  allow_pickle=True).tolist()
    print(merge_items(adm['timeseries']['codes'], set(
        [23634]), 123, keep_func=lambda x: True))
    # print(get_column_set([454,223900], map_itemid_index))

    # ## Making the map of features and itemids
    #
    # Here we define the map between features and itemids. Most features come from two data sources and we have to manually define the relationship between features and itemids and merge the data. We also assign the rank of column for each feature.

    # %%
    # derive 17 features from manually selected itemids
    # https://docs.google.com/spreadsheets/d/1e2KqLn3LTvcUwpSe5oE2ADwIEmUH9Xh54VADYVQ9mEQ/edit?ts=5960262a#gid=750248768
    feature_itemids = OrderedDict([
        ['gcsverbal', [723, 223900]],
        ['gcsmotor', [454, 223901]],
        ['gcseyes', [184, 220739]],
        #     ['glasgow_coma_scale', [454, 223900]],
        #     ['systolic_blood_pressure_abp_high_6', [6, 220050]],
        #     ['systolic_blood_pressure_abp_high_51', [51, 220050]],
        #     ['systolic_blood_pressure_abp_high_6701', [6701, 220050]],
        ['systolic_blood_pressure_abp_mean', [51, 442, 455, 6701, 220050, 220179]],
        #     ['systolic_blood_pressure_abp_high_mean', [6, 51, 6701, 220050]],
        #     ['systolic_blood_pressure_abp_high_max', [6, 51, 6701, 220050]],
        #     ['systolic_blood_pressure_abp_high_min', [6, 51, 6701, 220050]],
        #     ['systolic_blood_pressure_abp_low', [6]],
        #     ['systolic_blood_pressure_nbp_high', [455, 220179]],
        #     ['systolic_blood_pressure_nbp_low', []],
        ['heart_rate', [211, 220045]],
        ['body_temperature', {
            'f': [678, 223761],
            'c': [676, 223762]
        }],
        ['pao2', [50821]],
        ['fio2', [50816, 223835, 3420, 3422, 190]],
        #     ['pao2_fio2_ratio', [50821, 50816]],
        ['urinary_output', [40055,
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
        #     ['white_blood_cells_count_51300', [51300]],
        #     ['white_blood_cells_count_51301', [51301]],
        ['white_blood_cells_count_mean', [51300, 51301]],
        #     ['white_blood_cells_count_max', [51300, 51301]],
        #     ['white_blood_cells_count_min', [51300, 51301]],
        #     ['serum_bicarbonate_level_50803', [50803]],
        #     ['serum_bicarbonate_level_50804', [50804]],
        #     ['serum_bicarbonate_level_50802', [50802]],
        ['serum_bicarbonate_level_mean', [50882]],
        #     ['serum_bicarbonate_level_max', [50803, 50804, 50802]],
        #     ['serum_bicarbonate_level_min', [50803, 50804, 50802]],
        #     ['sodium_level_50824', [50824]],
        #     ['sodium_level_50983', [50983]],
        ['sodium_level_mean', [50824, 50983]],
        #     ['sodium_level_max', [50824, 50983]],
        #     ['sodium_level_min', [50824, 50983]],
        #     ['potassium_level_50822', [50822]],
        #     ['potassium_level_50971', [50971]],
        ['potassium_level_mean', [50822, 50971]],
        #     ['potassium_level_max', [50822, 50971]],
        #     ['potassium_level_min', [50822, 50971]],
        ['bilirubin_level', [50885]],
        #     ['type_of_admission', []],
        #     ['acquired_immunodeficiency_syndrome', []],
        #     ['metastatic_cancer', []],
        #     ['hematologic_malignancy', []],
        ['timestamp', []],
        ['aid', []]
    ])

    merge_funcs = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min
    }

    map_feature_colids = {}
    t = 0
    for key in feature_itemids.keys():
        map_feature_colids[key] = t
        t += 1
    map_feature_colids
    print(len(map_feature_colids))
    print(map_feature_colids)

    # %%
    np.save(os.path.join(RAWDIR, 'map_feature_colids.npy'), map_feature_colids)

    # ## Collect names of columns for verification

    # %%
    conn = getConnection()
    cur = conn.cursor()
    for feature, itemids in feature_itemids.items():
        if len(itemids) == 0:
            continue
        if type(itemids) == type({}):
            for key, value in itemids.items():
                sql = 'select itemid, label from mimiciii.d_items where itemid in ({0}) union all select itemid, label from mimiciii.d_labitems where itemid in ({0})'.format(
                    ','.join(list(map(str, value))))
                cur.execute(sql)
                res = cur.fetchall()
                print(feature + ' ' + key)
                for r in res:
                    print('{0},{1}'.format(r[0], r[1]))
                print()
        else:
            sql = 'select itemid, label from mimiciii.d_items where itemid in ({0}) union all select itemid, label from mimiciii.d_labitems where itemid in ({0})'.format(
                ','.join(list(map(str, itemids))))
            cur.execute(sql)
            res = cur.fetchall()
            print(feature)
            for r in res:
                print('{0},{1}'.format(r[0], r[1]))
            print()

    # ## Extract temporal features
    #
    # Since the number of temporal features is limited, we manually define the processing method for each feature in the following code.
    #
    # - body temperature: convert Farenheit to Celcius, use Celcius in priority in confliction
    # - urinary output: use the sum of all related itemids
    # - other features: use mean value when meeting confliction

    # %%

    # %%
    p = Pool(args.num_workers)
    for aid in valid_aids:
        p.apply_async(extract_adm, args=(aid, SOURCEDIR, PROCESSED_DB_DIR, RAWDIR,
                                         feature_itemids, map_itemid_index, map_feature_colids, merge_funcs, LABELDIR))
    p.close()
    p.join()

    def collect_admissions_with_more_than_hrs(hrs):
        processed_data_all = np.load(os.path.join(
            TARGETDIR, '%dhrs' % hrs, 'DB_merged_%dhrs.npy' % hrs), allow_pickle=True).tolist()
        processed_valid_aids = sorted([t[0][-1] for t in processed_data_all])

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

    # > 24hrs
    collect_admissions_with_more_than_hrs(24)

    # %%
    collect_admissions_with_more_than_hrs(48)
