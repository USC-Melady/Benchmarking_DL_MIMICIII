# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# # Valid Dataset
#
# Here we only keep admissions which are the first admissions of some patients and are of patients >= 15 years old.

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


def copy_valid_admissions(aid, admission_first_ids_set, cachedir, TARGETDIR):
    if aid in admission_first_ids_set:
        res = np.load(cachedir.joinpath('admdata', 'adm-%.6d.npy' % aid), allow_pickle=True).tolist()
        general = res['general']
        age = general[2]
        if age >= 15 * 365.25:
            np.save(os.path.join(TARGETDIR, 'adm-%.6d.npy' % aid), res)


# %%
def getValidDataset(args):
    cachedir = Path(args.cachedir)
    _adm_first = np.load(cachedir.joinpath('res/admission_first_ids.npy'), allow_pickle=True).tolist()
    admission_first_ids_list = _adm_first['admission_ids']

    admission_ids = [re.match(r'adm\-(\d+)\.npy', x)
                     for x in os.listdir(cachedir.joinpath('admdata'))]
    admission_ids = [int(x.group(1)) for x in admission_ids if x is not None]
    print(len(admission_ids), admission_ids[:10])

    admission_first_ids_set = set(admission_first_ids_list)
    admission_first_ids = [
        x for x in admission_ids if x in admission_first_ids_set]
    print(len(admission_first_ids), admission_first_ids[:10])

    # %%
    TARGETDIR = cachedir.joinpath('admdata_valid')
    if not os.path.exists(TARGETDIR):
        os.makedirs(TARGETDIR)

    # ## Store valid data
    #
    # We store all datafiles belonging to valid admission ids in a specific folder (../../Data/admdata_valid)

    # %%

    # p = Pool(args.num_workers)
    # for aid in admission_ids:
    #     p.apply_async(copy_valid_admissions, args=(
    #         aid, admission_first_ids_set, cachedir, TARGETDIR))
    # p.close()
    # p.join()
    for aid in tqdm(admission_ids):
        copy_valid_admissions(aid, admission_first_ids_set, cachedir, TARGETDIR)