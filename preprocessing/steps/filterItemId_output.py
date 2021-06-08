# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# # Filter Itemid Output
#
# This script is used for filtering itemids from TABLE OUTPUTEVENTS.
#
# 1. We check number of units of each itemid and choose the major unit as the target of unit conversion. In fact, for outputevents the units are the same - 'mL'.
# 2. In this step we do not apply any filtering to the data.
#
# ## Output
#
# 1. itemid of observations for outputevents.
# 2. unit of measurement for each itemid. Here we use None since no conversion is needed.

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
from multiprocessing import Pool
from pathlib import Path

from preprocessing.utils import getConnection


def filterItemId_output(args):
    conn = getConnection()
    cachedir = Path(args.cachedir)
    _adm = np.load(cachedir.joinpath('res/admission_ids.npy'),
                   allow_pickle=True).tolist()
    admission_ids = _adm['admission_ids']
    admission_ids_txt = _adm['admission_ids_txt']

    db = np.load(cachedir.joinpath('res/itemids.npy'),
                 allow_pickle=True).tolist()
    input_itemid = db['input']
    output_itemid = db['output']
    chart_itemid = db['chart']
    lab_itemid = db['lab']
    microbio_itemid = db['microbio']
    prescript_itemid = db['prescript']

    # %%
    cur = conn.cursor()
    cur.execute('select distinct valueuom from mimiciii.outputevents')

    # All records have the same unit. Therefore just keep all itemids.

    # %%
    valid_output = output_itemid
    np.save(cachedir.joinpath('res/filtered_output.npy'),
            {'id': valid_output, 'unit': None})
