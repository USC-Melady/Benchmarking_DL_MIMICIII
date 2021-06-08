
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
import re
import traceback
import shutil

from preprocessing.utils import getConnection
from preprocessing.utils import parseUnitsMap
from preprocessing.utils import parseNum
from preprocessing.utils import sparsify


# %%
def run_necessary_sqls(args):
    conn = getConnection()
    cur = conn.cursor()
    working_dir = './mimic-code/'

    # prepare necessary materialized views

    sqlfilelist = [
        'concepts/echo-data.sql',
        'concepts/ventilation-durations.sql',
        'concepts/firstday/vitals-first-day.sql',
        'concepts/firstday/urine-output-first-day.sql',
        'concepts/firstday/ventilation-first-day.sql',
        'concepts/firstday/gcs-first-day.sql',
        'concepts/firstday/labs-first-day.sql',
        'concepts/firstday/blood-gas-first-day.sql',
        'concepts/firstday/blood-gas-first-day-arterial.sql',
        'concepts_48/echo-data.sql',
        'concepts_48/ventilation-durations.sql',
        'concepts_48/firstday/vitals-first-day.sql',
        'concepts_48/firstday/urine-output-first-day.sql',
        'concepts_48/firstday/ventilation-first-day.sql',
        'concepts_48/firstday/gcs-first-day.sql',
        'concepts_48/firstday/labs-first-day.sql',
        'concepts_48/firstday/blood-gas-first-day.sql',
        'concepts_48/firstday/blood-gas-first-day-arterial.sql'
    ]

    for sqlfile in sqlfilelist:
        pstr = os.path.join(working_dir, sqlfile)
        if not os.path.exists(pstr):
            print(pstr)

    for sqlfile in sqlfilelist:
        print('executing {0}...'.format(sqlfile))
        with open(os.path.join(working_dir, sqlfile), 'r') as f:
            sql = f.read()
            cur.execute(sql)
            conn.commit()
        print('finish executing {0}!'.format(sqlfile))

    # prepare time series

    conn = getConnection()
    cur = conn.cursor()
    working_dir = 'preprocessing/sql_gen_17features_ts/'

    sqlfilelist = [
        'gen_gcs_ts.sql',
        'gen_lab_ts.sql',
        'gen_pao2_fio2.sql',
        'gen_urine_output_ts.sql',
        'gen_vital_ts.sql',
        'gen_17features_first24h.sql',
        'gen_17features_first48h.sql',
        'create_indices.sql'
    ]

    for sqlfile in sqlfilelist:
        pstr = os.path.join(working_dir, sqlfile)
        if not os.path.exists(pstr):
            print(pstr)

    for sqlfile in sqlfilelist:
        print('executing {0}...'.format(sqlfile))
        with open(os.path.join(working_dir, sqlfile), 'r') as f:
            sql = f.read()
            cur.execute(sql)
            conn.commit()
        print('finish executing {0}!'.format(sqlfile))
