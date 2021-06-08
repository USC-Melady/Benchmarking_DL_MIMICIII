#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def get_avg_99plus_features_raw_hrs(args, hrs):
    HRS = hrs
    cachedir = Path(args.cachedir)
    TARGETDIR = cachedir.joinpath('admdata_99p')
    HRDIR = os.path.join(TARGETDIR, '%dhrs_raw' % HRS)
    # HRDIR = os.path.join(TARGETDIR, '%dhrs' % HRS)
    RESDIR = os.path.join(HRDIR, 'non_series')
    SERIALDIR = os.path.join(HRDIR, 'series')

    if not os.path.exists(RESDIR):
        os.makedirs(RESDIR)

    hrs_mean = np.load(os.path.join(RESDIR, 'tsmean_%dhrs.npz' % HRS), allow_pickle=True)
    hrs_mean_array = hrs_mean['hrs_mean_array']
    hrs_mean_labels = hrs_mean['hrs_mean_labels']

    INPUTFILEPATH = os.path.join(RESDIR, 'input.csv')
    ress = hrs_mean_array
    with open(INPUTFILEPATH, 'w') as f:
        for res in ress:
            f.write(','.join(list(map(lambda x: str(x) if x is not None else '', res))) + '\n')


    # In[3]:


    len(ress)


    # In[4]:


    # labels
    adm_labels_all = np.load(os.path.join(HRDIR, 'ADM_LABELS_%dhrs.npy' % HRS), allow_pickle=True)
    with open(os.path.join(RESDIR, 'output.csv'), 'w') as f:
        for res in adm_labels_all:
            f.write(','.join(list(map(str, res))) + '\n')


    # In[5]:


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

    data_all = np.load(os.path.join(HRDIR, 'DB_merged_%dhrs.npy' % HRS), allow_pickle=True).tolist()
    valid_aids = [t[0][-1] for t in data_all]
    print(len(valid_aids))
    mv_flag = np.array([valid_aid in mv_admset for valid_aid in valid_aids])
    np.save(os.path.join(RESDIR, 'mv_flag.npy'), mv_flag)

    # input mv
    inputarray = np.genfromtxt(os.path.join(RESDIR, 'input.csv'), delimiter=',')[mv_flag]
    # output mv
    outputlabels = np.genfromtxt(os.path.join(RESDIR, 'output.csv'), delimiter=',')[mv_flag].astype(int)
    # save!
    np.savetxt(os.path.join(RESDIR, 'input_mv.csv'), inputarray, delimiter=',')
    np.savetxt(os.path.join(RESDIR, 'output_mv.csv'), outputlabels, delimiter=',')
    # input cv
    inputarray = np.genfromtxt(os.path.join(RESDIR, 'input.csv'), delimiter=',')[~mv_flag]
    # output cv
    outputlabels = np.genfromtxt(os.path.join(RESDIR, 'output.csv'), delimiter=',')[~mv_flag].astype(int)
    # save!
    np.savetxt(os.path.join(RESDIR, 'input_cv.csv'), inputarray, delimiter=',')
    np.savetxt(os.path.join(RESDIR, 'output_cv.csv'), outputlabels, delimiter=',')


    # In[6]:


    # def gen_file_for_r(FOLDSPATH, FOLDSOUTPATH, RESDIR, inputfilename, outputfilename):
    #     if not os.path.exists(FOLDSOUTPATH):
    #         os.makedirs(FOLDSOUTPATH)
    #     inputarray = np.genfromtxt(os.path.join(RESDIR, inputfilename), delimiter=',')
    #     outputarray = np.genfromtxt(os.path.join(RESDIR, outputfilename), delimiter=',')
    #     for t in range(len(adm_labels_all[0])):
    #         folds = np.load(os.path.join(FOLDSPATH, '5-folds.npz'), allow_pickle=True)['folds_ep_mor'][t][0]
    #         for fi, f in enumerate(folds):
    #             train, valid, test = f[0], f[1], f[2]
    #             train = np.concatenate((train, valid))
    #             Xtrain = inputarray[train, :]
    #             train_mean = np.nanmean(Xtrain, axis=0)
    #             train_std = np.nanstd(Xtrain, axis=0)
    #             newinput = np.copy(inputarray)
    #             for l in range(newinput.shape[0]):
    #                 newinput[l,:] = (newinput[l,:] - train_mean) / train_std
    #             newinput[np.isinf(newinput)] = 0
    #             newinput[np.isnan(newinput)] = 0
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'input_train_%d_%d.csv' % (fi, t)), newinput[train], delimiter=',')
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'output_train_%d_%d.csv' % (fi, t)), outputarray[train], delimiter=',')
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'input_test_%d_%d.csv' % (fi, t)), newinput[test], delimiter=',')
    #             np.savetxt(os.path.join(FOLDSOUTPATH, 'output_test_%d_%d.csv' % (fi, t)), outputarray[test], delimiter=',')
    #             print(os.path.join(FOLDSOUTPATH, 'input_%d_%d.csv' % (fi, t)))
                
    # gen_file_for_r(
    #     SERIALDIR,
    #     os.path.join(RESDIR, 'folds'),
    #     RESDIR,
    #     'input.csv',
    #     'output.csv'
    # )

    # gen_file_for_r(
    #     os.path.join(SERIALDIR, 'cv'),
    #     os.path.join(RESDIR, 'folds', 'cv'),
    #     RESDIR,
    #     'input_cv.csv',
    #     'output_cv.csv'
    # )


def get_avg_99plus_features_raw(args):
    get_avg_99plus_features_raw_hrs(args, 24)
    get_avg_99plus_features_raw_hrs(args, 48)