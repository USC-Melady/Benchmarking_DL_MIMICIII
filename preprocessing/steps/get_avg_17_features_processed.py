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


# In[2]:

def get_avg_17_features_processed_hrs(args, hrs):
    cachedir = Path(args.cachedir)
    HRS = hrs
    # HRS = 48
    TARGETDIR = cachedir.joinpath('admdata_17f')
    HRDIR = os.path.join(TARGETDIR, '%dhrs' % HRS)
    RESDIR = os.path.join(HRDIR, 'non_series')
    SERIALDIR = os.path.join(HRDIR, 'series')

    if not os.path.exists(RESDIR):
        os.makedirs(RESDIR)

    # In[4]:

    data_all = np.load(os.path.join(HRDIR, 'DB_merged_%dhrs.npy' %
                                    HRS), allow_pickle=True).tolist()
    valid_aids = [t[0][-1] for t in data_all]
    print(len(valid_aids))
    # print(valid_aids)

    admtype_map = {
        'scheduledsurgical': 1,
        'unscheduledsurgical': 2,
        'medical': 0
    }

    # In[ ]:

    def fetch_aid(aid):
        conn = getConnection()
        cur = conn.cursor()
        sql = 'select * from mengcz_17features_first{0}h where hadm_id={1}'.format(
            HRS, aid)
        cur.execute(sql)
        res = cur.fetchone()
        if res is None:
            return None
        res = list(res)[5:]
        if res[-4] is None:
            res[-4] = 0
        if res[-3] is None:
            res[-3] = 0
        if res[-2] is None:
            res[-2] = 0
        res[-1] = admtype_map[res[-1].lower()]
        return res

    # p = Pool(16)
    # ress = p.map(fetch_aid, valid_aids)
    # p.close()
    # p.join()
    ress = []
    for aid in tqdm(valid_aids):
        ress.append(fetch_aid(aid))

    assert len(ress) == len(valid_aids)
    assert len([1 for x in ress if x is None]) == 0

    INPUTFILEPATH = os.path.join(RESDIR, 'input.csv')
    with open(INPUTFILEPATH, 'w') as f:
        for res in ress:
            f.write(
                ','.join(list(map(lambda x: str(x) if x is not None else '', res))) + '\n')

    # In[8]:

    # labels
    adm_labels_all = np.load(os.path.join(
        HRDIR, 'ADM_LABELS_%dhrs.npy' % HRS), allow_pickle=True)
    with open(os.path.join(RESDIR, 'output.csv'), 'w') as f:
        for res in adm_labels_all:
            f.write(','.join(list(map(str, res))) + '\n')

    # In[9]:

    # sapsii_scores = np.load(os.path.join(
    #     RESDIR, 'sapsii.npz'), allow_pickle=True)['sapsii']
    # sapsii_subscores = sapsii_scores[:, 5:].astype(np.float64)
    # sapsii_subscores[np.isnan(sapsii_subscores)] = 0
    # np.savetxt(os.path.join(RESDIR, 'input_sapsiisubscores.csv'),
    #            sapsii_subscores, delimiter=',')

    # In[10]:

    sql = 'select distinct hadm_id from mimiciii.icustays where dbsource = \'metavision\' '
    sql += 'UNION select distinct hadm_id from mimiciii.transfers where dbsource = \'metavision\''
    conn = getConnection()
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()

    admission_ids = []
    for r in res:
        admission_ids.append(r[0])
    mv_admset = set(admission_ids)
    mv_flag = np.array([valid_aid in mv_admset for valid_aid in valid_aids])
    np.save(os.path.join(RESDIR, 'mv_flag.npy'), mv_flag)

    # input mimicii
    inputarray = np.genfromtxt(os.path.join(
        RESDIR, 'input.csv'), delimiter=',')[mv_flag]
    # output mimicii
    outputlabels = np.genfromtxt(os.path.join(RESDIR, 'output.csv'), delimiter=',')[
        mv_flag].astype(int)
    # save!
    np.savetxt(os.path.join(RESDIR, 'input_mv.csv'), inputarray, delimiter=',')
    np.savetxt(os.path.join(RESDIR, 'output_mv.csv'),
               outputlabels, delimiter=',')

    # input_trans = np.genfromtxt(os.path.join(
    #     RESDIR, 'input_sapsiisubscores.csv'), delimiter=',')[mv_flag]
    # np.savetxt(os.path.join(RESDIR, 'input_sapsiisubscores_mv.csv'),
    #            input_trans, delimiter=',')

    # input mimicii
    inputarray = np.genfromtxt(os.path.join(
        RESDIR, 'input.csv'), delimiter=',')[~mv_flag]
    # output mimicii
    outputlabels = np.genfromtxt(os.path.join(RESDIR, 'output.csv'), delimiter=',')[
        ~mv_flag].astype(int)
    # save!
    np.savetxt(os.path.join(RESDIR, 'input_cv.csv'), inputarray, delimiter=',')
    np.savetxt(os.path.join(RESDIR, 'output_cv.csv'),
               outputlabels, delimiter=',')

    # input_trans = np.genfromtxt(os.path.join(
    #     RESDIR, 'input_sapsiisubscores.csv'), delimiter=',')[~mv_flag]
    # np.savetxt(os.path.join(RESDIR, 'input_sapsiisubscores_cv.csv'),
    #            input_trans, delimiter=',')

    # ## Generate input files for R scripts
    #
    # Since it is not convenient to do the normalization in R, here we finish the normalization and generate one input file for each fold. These files are only used to evaluate the performance of SuperLearner(R version) on mortality prediction tasks.
    # - input_train_F_T.csv: features of training set in the Fth fold on the Tth mortality prediction task
    # - output_train_F_T.csv: mortality labels of training set in the Fth fold on the Tth mortality prediction task
    # - input_test_F_T.csv: features of test set in the Fth fold on the Tth mortality prediction task
    # - output_test_F_T.csv: mortality labels of test set in the Fth fold on the Tth mortality prediction task

    # In[11]:

    # FOLDSPATH = '../../Data/admdata_17f/24hrs/series/'
    # FOLDSOUTPATH = '../../Data/admdata_17f/24hrs/non_series/folds'

    # def gen_file_for_r(FOLDSPATH, FOLDSOUTPATH, RESDIR, inputfilename, outputfilename):
    #     if not os.path.exists(FOLDSOUTPATH):
    #         os.makedirs(FOLDSOUTPATH)
    #     inputarray = np.genfromtxt(os.path.join(
    #         RESDIR, inputfilename), delimiter=',')
    #     outputarray = np.genfromtxt(os.path.join(
    #         RESDIR, outputfilename), delimiter=',')
    #     for t in range(len(adm_labels_all[0])):
    #         folds = np.load(os.path.join(FOLDSPATH, '5-folds.npz'),
    #                         allow_pickle=True)['folds_ep_mor'][t][0]
    #         for fi, f in enumerate(folds):
    #             train, valid, test = f[0], f[1], f[2]
    #             train = np.concatenate((train, valid))
    #             Xtrain = inputarray[train, :]
    #             train_mean = np.nanmean(Xtrain, axis=0)
    #             train_std = np.nanstd(Xtrain, axis=0)
    #             newinput = np.copy(inputarray)
    #             for l in range(newinput.shape[0]):
    #                 newinput[l, :] = (newinput[l, :] - train_mean) / train_std
    #             newinput[np.isinf(newinput)] = 0
    #             newinput[np.isnan(newinput)] = 0
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'input_train_%d_%d.csv' % (
    #                 fi, t)), newinput[train], delimiter=',')
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'output_train_%d_%d.csv' % (
    #                 fi, t)), outputarray[train], delimiter=',')
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'input_test_%d_%d.csv' % (
    #                 fi, t)), newinput[test], delimiter=',')
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'output_test_%d_%d.csv' % (
    #                 fi, t)), outputarray[test], delimiter=',')
    #             print(os.path.join(FOLDSOUTPATH, 'input_%d_%d.csv' % (fi, t)))

    # # In[ ]:

    # # for 17 features on mimiciii
    # gen_file_for_r(
    #     SERIALDIR,
    #     os.path.join(RESDIR, 'folds'),
    #     RESDIR,
    #     'input.csv',
    #     'output.csv'
    # )

    # # for subscores on mimiciii
    # gen_file_for_r(
    #     SERIALDIR,
    #     os.path.join(RESDIR, 'folds_sapsiiscores'),
    #     RESDIR,
    #     'input_sapsiisubscores.csv',
    #     'output.csv'
    # )

    # # In[ ]:

    # # for 17 features on mimicii
    # gen_file_for_r(
    #     os.path.join(SERIALDIR, 'cv'),
    #     os.path.join(RESDIR, 'folds', 'cv'),
    #     RESDIR,
    #     'input_cv.csv',
    #     'output_cv.csv'
    # )

    # # for subscores on mimicii
    # gen_file_for_r(
    #     os.path.join(SERIALDIR, 'cv'),
    #     os.path.join(RESDIR, 'folds_sapsiiscores', 'cv'),
    #     RESDIR,
    #     'input_sapsiisubscores_cv.csv',
    #     'output_cv.csv'
    # )


def get_avg_17_features_processed(args):
    get_avg_17_features_processed_hrs(args, 24)
    get_avg_17_features_processed_hrs(args, 48)
