# # Create Admission List
#
# This script is used for creating a list of admission ids. We only keep admissions that are the first admissions of their patients.

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

from preprocessing.utils import getConnection


# ## Select all admissions
#
# We collect all admission_ids from TABLE ICUSTAYS and TABLE TRANSFERS

# %%
# Select from icustay and transfer table
def createAdmissionList(args):
    print('0_createAdmissionList: Select all admissions from TABLE ICUSTAYS and TABLE TRANSFERS. Also collect admissions which are the first admissions of their patients.')

    conn = getConnection()
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS admission_ids')
    cur.execute('create table if not exists admission_ids as (select distinct hadm_id from mimiciii.icustays union select distinct hadm_id from mimiciii.transfers)')
    conn.commit()

    cur = conn.cursor()
    cur.execute('ALTER TABLE admission_ids ADD CONSTRAINT hadm_id PRIMARY KEY (hadm_id)')
    conn.commit()

    cur = conn.cursor()
    cur.execute('select * from admission_ids')
    res = cur.fetchall()

    admission_ids = [r[0] for r in res]
    admission_ids_txt = ','.join(map(str, admission_ids))

    # %%
    # number of admission id
    print('#admissions = ', len(admission_ids))

    # %%
    resdir = os.path.join(args.cachedir, 'res')
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    # save to admission_ids.npy
    tosave = {'admission_ids': admission_ids,
              'admission_ids_txt': admission_ids_txt}
    np.save(os.path.join(resdir, 'admission_ids.npy'), tosave)

    # Make sure that there is no duplication in admission_ids.

    # %%
    try:
        assert len(admission_ids) == len(set(admission_ids))
    except AssertionError:
        sys.exit('Duplications in admission_ids!')

    # ## Remove non-first admissions
    #
    # We remove all admissions which are not the first admissions of some patients in order to prevent possible information leakage, which will happen when multiple admissions of the same patient occur in training set and test set simultaneously.

    # %%
    # get the list of admission ids which is the first admission of the subject
    conn = getConnection()
    cur = conn.cursor()

    # fixed by https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/issues/12#issuecomment-680422181 to ensure that "distinct" retrives the first admission
#     cur.execute('select hadm_id from admission_ids where hadm_id in (select distinct on (subject_id) hadm_id from (select * from mimiciii.admissions order by admittime) tt)')
    cur.execute('select hadm_id from admission_ids where hadm_id in (select distinct on (subject_id) hadm_id from mimiciii.admissions order by subject_id,admittime)')
    res = cur.fetchall()

    admission_first_ids = [r[0] for r in res]
    admission_first_ids_txt = ','.join(list(map(str, admission_first_ids)))
    tosave = {'admission_ids': admission_first_ids,
              'admission_ids_txt': admission_first_ids_txt}
    np.save(os.path.join(resdir, 'admission_first_ids.npy'), tosave)
    print('#first admissions:', len(admission_first_ids))

    print('Finished 0_createAdmissionList!')
    print()
