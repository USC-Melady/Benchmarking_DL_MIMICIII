# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# # Filter Itemid Microbio
#
# This script is used for filtering itemids from TABLE MICROBIOLOGYEVENTS.
#
# 1. We only need to keep all itemids from MICROBIOLOGYEVENTS!
# 2. No need for unit conversion!
#
# ## Output
#
# 1. itemid of observations for microbiologyevents.

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
from pathlib import Path

from preprocessing.utils import getConnection

def filterItemId_microbio(args):
    conn = getConnection()

    cachedir = Path(args.cachedir)
    _adm = np.load(cachedir.joinpath('res/admission_ids.npy'), allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    db = np.load(cachedir.joinpath('res/itemids.npy'), allow_pickle=True).tolist()
    input_itemid = db['input']
    output_itemid = db['output']
    chart_itemid = db['chart']
    lab_itemid = db['lab']
    microbio_itemid = db['microbio']
    prescript_itemid = db['prescript']


    # %%
    valid_microbio = microbio_itemid
    np.save(cachedir.joinpath('res/filtered_microbio.npy'), {'id': valid_microbio, 'unit': None})
