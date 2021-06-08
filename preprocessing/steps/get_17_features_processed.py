# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# # Get 17 processed features
#
# This script is used for get 17 processed features from the database, since the generation of 17 processed features is done with SQL and the results are stored in the database.
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
from tqdm import tqdm

from preprocessing.utils import getConnection
from preprocessing.utils import parseUnitsMap
from preprocessing.utils import parseNum
from preprocessing.utils import sparsify


def gen_features_aid(aid, queryd, features, admtype_map, LABELDIR, RAWDIR):
    conn = getConnection()
    cur = conn.cursor()
    cur.execute(
        'select intime from mimiciii.mengcz_17features_first24h where hadm_id={0}'.format(aid))
    admission = cur.fetchone()
    if admission is None or admission[0] is None:
        return None
    admittime = admission[0]

    # time series
    time_series = []
    sqls = []

    for tablename, queryl in queryd.items():
        sql = 'select charttime, {0} from {1} where hadm_id={2}'.format(
            ','.join([q[0] for q in queryl]),
            tablename,
            str(aid)
        )
        cur = conn.cursor()
        cur.execute(sql)
        res = cur.fetchall()
        if res is None:
            return None
        cns = [q[1] for q in queryl]
        for rec in res:
            values = list(rec)[1:]
            if rec[0] is not None:
                timestampsec = (rec[0] - admittime).total_seconds()
                for value, cn in zip(values, cns):
                    if value is not None:
                        time_series.append((timestampsec, cn, value))
#     for featurename, table_col in features['ts'].items():
#         sql = 'select charttime, {0} as colnum, {1} as valuenum from {2} where hadm_id={3}'.format(
#             feature_col_map[featurename],
#             table_col[1],
#             table_col[0],
#             str(aid)
#         )
#         sqls.append(sql)
#     sqls = ' union all '.join(sqls)
#     cur = conn.cursor()
#     cur.execute(sqls)
#     res = cur.fetchall()
#     if res is None:
#         return None
#     for values in res:
#         if values is None:
#             continue
#         if values[0] is None or values[2] is None:
#             continue
#         time_series.append(((values[0] - admittime).total_seconds(), values[1], values[2]))

    if len(time_series) == 0:
        return None

    time_col_id = len(features['ts'])
    aid_col_id = time_col_id + 1

    timeset = sorted(list(set([v[0] for v in time_series])))
    timestampmap = {}
    for t, timestamp in enumerate(timeset):
        timestampmap[timestamp] = t
    time_series_sparse = [(timestampmap[ts[0]], ts[1], ts[2])
                          for ts in time_series]
    for t, timestamp in enumerate(timeset):
        time_series_sparse.append((t, time_col_id, timestamp))
    for t in range(len(timeset)):
        time_series_sparse.append((t, aid_col_id, aid))
    # time_series_sparse

    # admission features
    cur = conn.cursor()
    sql = 'select age, coalesce(AIDS, 0), coalesce(HEM, 0), coalesce(METS, 0), AdmissionType from mengcz_17features_first24h where hadm_id={0}'.format(
        aid)
    cur.execute(sql)
    res = cur.fetchone()
    if res is None:
        return None
    adm_features = (float(res[0]) * 365.242, res[1],
                    res[2], res[3], admtype_map[res[4].lower()])

    # admission labels
#     admres = np.load(os.path.join(SOURCEDIR, 'adm-%.6d.npy' % aid)).tolist()
#     general = admres['general']
#     mortal, die24, die24_48, die48_72, die30days, die1year = general[4], general[6], general[7], general[8], general[9], general[10]
#     adm_labels = (mortal, die24, die24_48, die48_72, die30days, die1year)
    admlabel = np.load(os.path.join(LABELDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
    adm_labels = (
        admlabel['mor'],
        admlabel['mor24'],
        admlabel['mor48'],
        admlabel['mor72'],
        admlabel['mor30d'],
        admlabel['mor1y'],
    )

    try:
        res = {
            'serial_features': {
                'codes': time_series_sparse,
                'timestep': len(timeset),
                'features': aid_col_id + 1,
                'timelength': timeset[-1] - timeset[0]
            },
            'adm_features': adm_features,
            'adm_labels': adm_labels
        }
        np.save(os.path.join(
            RAWDIR, 'adm-{0}.npy'.format(str('%.6d' % aid))), res)
    #         print('finished {0}!'.format(aid))
        return res
    except:
        print('fail at {0}!'.format(aid))
        return None


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


def extract_data(aid, hrs, RAWDIR, SOURCEDIR, LABELDIR):
    admres = np.load(os.path.join(RAWDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
    if check_adm_hrs_pass(admres, hrs):
        ori_admres = np.load(os.path.join(
            SOURCEDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
        admlabel = np.load(os.path.join(
            LABELDIR, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
        adm_labels = (
            admlabel['mor'],
            admlabel['mor24'],
            admlabel['mor48'],
            admlabel['mor72'],
            admlabel['mor30d'],
            admlabel['mor1y'],
        )
        return (
            extract_time_series(admres),
            ori_admres['icd9'],
            ori_admres['general'],
            admres['adm_features'],
            adm_labels
        )
    else:
        return None


# %%
def get_17_features_processed(args):
    cachedir = Path(args.cachedir)
    # get all valid admission ids: age > 15
    SOURCEDIR = cachedir.joinpath('admdata_valid')
    TARGETDIR = cachedir.joinpath('admdata_17f')
    LABELDIR = cachedir.joinpath('admdata_timelabels')

    if not os.path.exists(TARGETDIR):
        os.makedirs(TARGETDIR)

    valid_aids = [re.match(r'adm\-(\d+)\.npy', x)
                  for x in os.listdir(SOURCEDIR)]
    valid_aids = sorted([int(x.group(1)) for x in valid_aids if x is not None])
    print(len(valid_aids), valid_aids[:10])

    # ## Set the map between feature name, table name and column name
    #
    # Here we manually set the map between feature name, table name and column name as
    # ```
    # [feature name]: [[table name], [column name]]
    # ```

    # %%
    # 17 features: features used in calculating SAPS II score
    # Here mean/max/min is done for values with the same aid and the same timestamp, only for solving conflict
    features = OrderedDict([
        ['ts', OrderedDict([
            ['glasgow_coma_scale', ['mengcz_glasgow_coma_scale_ts', 'GCS']],
            ['systolic_blood_pressure', ['mengcz_vital_ts', 'SysBP_Mean']],
            ['heart_rate', ['mengcz_vital_ts', 'HeartRate_Mean']],
            ['body_temperature', ['mengcz_vital_ts', 'TempC_Mean']],
            #         ['pao2_fio2_ratio', ['mengcz_pao2fio2_ts', 'PaO2FiO2']],
            ['pao2', ['mengcz_pao2fio2_ts', 'PO2']],
            ['fio2', ['mengcz_pao2fio2_ts', 'FIO2']],
            ['urinary_output', ['mengcz_urine_output_ts', 'UrineOutput']],
            ['serum_urea_nitrogen_level', ['mengcz_labs_ts', 'BUN_min']],
            ['white_blood_cells_count', ['mengcz_labs_ts', 'WBC_min']],
            ['serum_bicarbonate_level', ['mengcz_labs_ts', 'BICARBONATE_min']],
            ['sodium_level', ['mengcz_labs_ts', 'SODIUM_min']],
            ['potassium_level', ['mengcz_labs_ts', 'POTASSIUM_min']],
            ['bilirubin_level', ['mengcz_labs_ts', 'BILIRUBIN_min']],
        ])],
        ['static', OrderedDict([
            ['age', ['mengcz_17features_first24h', 'age']],
            ['aids', ['mengcz_17features_first24h', 'AIDS']],
            ['hem', ['mengcz_17features_first24h', 'HEM']],
            ['mets', ['mengcz_17features_first24h', 'METS']],
            ['admission_type', ['mengcz_17features_first24h', 'AdmissionType']],
        ])]
    ])

    # %%
    feature_col_list = list(features['ts'].keys()) + \
        list(features['static'].keys())
    feature_col_map = OrderedDict()
    for t, feature in enumerate(feature_col_list):
        feature_col_map[feature] = t
    feature_col_map

    # ## Extract features from database
    #
    # For each admission id, we extract 17 processed features from the database and store a file for each admission id in folder processed_db.

    # %%
    admtype_map = {
        'scheduledsurgical': 1,
        'unscheduledsurgical': 2,
        'medical': 0
    }

    RAWDIR = os.path.join(TARGETDIR, 'processed_db')
    if not os.path.exists(RAWDIR):
        os.makedirs(RAWDIR)

    queryd = {}
    for featurename, table_col in features['ts'].items():
        tn = table_col[0]
        cn = table_col[1]
        try:
            queryd[tn].append((cn, feature_col_map[featurename]))
        except:
            queryd[tn] = [(cn, feature_col_map[featurename])]

    print(queryd)

    # %%
    # p = Pool(args.num_workers)
    # for aid in valid_aids:
    #     p.apply_async(gen_features_aid, args=(
    #         aid, queryd, features, admtype_map, LABELDIR, RAWDIR))
    # p.close()
    # p.join()
    for aid in tqdm(valid_aids):
    # for aid in [197331, 190132, 184817]:
        gen_features_aid(aid, queryd, features, admtype_map, LABELDIR, RAWDIR)

    # ## Generate input files for sampling and imputation
    #
    # After this step, we get 5 input files needed for sampling and imputation.
    #
    # ## Select admissions with > xxhrs records
    #
    # We only keep admissions with record length > 24/48 hrs.

    def collect_admissions_with_more_than_hrs(hrs):
        raw_aids = [re.match(r'adm\-(\d+)\.npy', x)
                    for x in os.listdir(RAWDIR)]
        raw_aids = sorted([int(x.group(1)) for x in raw_aids if x is not None])
        HRDIR = os.path.join(TARGETDIR, '%dhrs' % hrs)
        if not os.path.exists(HRDIR):
            os.makedirs(HRDIR)

        p = Pool(args.num_workers)
        collec = [p.apply_async(extract_data, args=(
            aid, hrs, RAWDIR, SOURCEDIR, LABELDIR)) for aid in raw_aids]
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
    # > 48hrs
    collect_admissions_with_more_than_hrs(48)
