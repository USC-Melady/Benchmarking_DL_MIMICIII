# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# # Collect mortality labels
#
# We use this script to calculate mortality labels and store them in folder './admdata_times'. Labels generated here will be used in later steps of pre-processing.

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
from pathlib import Path
from tqdm import tqdm

from preprocessing.utils import getConnection
from preprocessing.utils import parseUnitsMap
from preprocessing.utils import parseNum
from preprocessing.utils import sparsify


def collect_timestamps(aid, timedir):
    conn = getConnection()
    cur = conn.cursor()
    sql = 'select subject_id, admittime, dischtime, deathtime from mimiciii.admissions where hadm_id={0}'.format(
        aid)
    cur.execute(sql)
    res = cur.fetchone()
    subject_id = res[0]
    admittime, dischtime, deathtime = res[1], res[2], res[3]
    sql = 'select dob, dod from mimiciii.patients where subject_id={0}'.format(
        subject_id)
    cur.execute(sql)
    res = cur.fetchone()
    dob, dod = res[0], res[1]
    sql = 'select intime, outtime from mimiciii.icustays where hadm_id={0} order by intime'.format(
        aid)
    cur.execute(sql)
    icutimepairs = cur.fetchall()
    data = {
        'dob': dob,
        'dod': dod,
        'admittime': admittime,
        'dischtime': dischtime,
        'deathtime': deathtime,
        'icustays': icutimepairs
    }
    np.save(os.path.join(timedir, 'adm-%.6d.npy' % aid), data)


def parse_labels(aid, timedir, timelabeldir):
    times = np.load(os.path.join(timedir, 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
    dob = times['dob']
    dod = times['dod']
    admittime = times['admittime']
    dischtime = times['dischtime']
    deathtime = times['deathtime']
    icustays = times['icustays']
    mor, mor24, mor48, mor72, mor30d, mor1y = 0, 0, 0, 0, 0, 0
    # choose starttime, here choose first icustay time in priority
    try:
        starttime = icustays[0][0]
    except:
        starttime = admittime
    if starttime is None:
        data = {
            'mor': None,
            'mor24': None,
            'mor48': None,
            'mor72': None,
            'mor30d': None,
            'mor1y': None
        }
        np.save(os.path.join(timelabeldir, 'adm-%.6d.npy' % aid), None)
        return
    # generate labels
    try:
        mor = int(deathtime is not None)
        assert mor == 1
        tlen = (deathtime - starttime).total_seconds()
        mor24 = int(tlen <= 24 * 60 * 60)
        mor48 = int(tlen <= 48 * 60 * 60)
        mor72 = int(tlen <= 72 * 60 * 60)
    except:
        pass
    try:
        livelen = (dod - dischtime).total_seconds()
        mor30d = int(livelen <= 30 * 24 * 60 * 60)
        mor1y = int(livelen <= 365.245 * 24 * 60 * 60)
    except:
        pass
    data = {
        'mor': mor,
        'mor24': mor24,
        'mor48': mor48,
        'mor72': mor72,
        'mor30d': mor30d,
        'mor1y': mor1y
    }
    np.save(os.path.join(timelabeldir, 'adm-%.6d.npy' % aid), data)


def collect_mortality_labels(args):
    cachedir = Path(args.cachedir)
    admdir = cachedir.joinpath('admdata')
    admaids = [re.match(r'adm\-(\d+)\.npy', x) for x in os.listdir(admdir)]
    admaids = sorted([int(x.group(1)) for x in admaids if x is not None])

    # %%

    # ## Generate mortality labels
    #
    # Here we collect all timestamps related to mortality labels.
    # Situations when the labels should be 1:
    # - in-hospital mortality: deathtime is not null
    # - 48/72 mortality: deathtime - icuintime <= 48/72hrs
    # - 30d/1yr mortality: dod - dischtime <= 30d/1yr

    # %%
    # Here we collect all timestamps related to our labels
    # we need: dob, dod, admittime, first_icuintime
    # admissions: admittime, dischtime, deathtime
    # patients: dob, dod
    # icustays: intime, outtime
    timedir = cachedir.joinpath('admdata_times')
    if not os.path.exists(timedir):
        os.makedirs(timedir)

    timelabeldir = cachedir.joinpath('admdata_timelabels')
    if not os.path.exists(timelabeldir):
        os.makedirs(timelabeldir)

    # %%
    # p = Pool(args.num_workers)
    # for aid in admaids:
    #     p.apply_async(collect_timestamps, args=(aid, timedir))
    # p.close()
    # p.join()

    # p = Pool(args.num_workers)
    # for aid in admaids:
    #     p.apply_async(parse_labels, args=(aid, timedir, timelabeldir))
    # p.close()
    # p.join()

    for aid in tqdm(admaids):
        collect_timestamps(aid, timedir)
        parse_labels(aid, timedir, timelabeldir)
