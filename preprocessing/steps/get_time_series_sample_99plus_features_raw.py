from __future__ import print_function
from copy import deepcopy
import math
import os
import sys
from pathlib import Path

import numpy as np
from preprocessing.steps.get_time_series_sample_17_features_processed import try_making_splits, make_splits
from preprocessing.utils import getConnection


# def get_icd9_subcat_label(icd9_str):
#     ss = icd9_str.split('.')[0]
#     idx_lb = max(np.where(ss >= subcat_lbs)[0])
#     idx_ub = min(np.where(ss[:4] <= subcat_ubs)[0])
#     if idx_lb != idx_ub:
#         print(idx_lb, idx_ub, icd9_str, ss)
#     #assert idx_lb == idx_ub
#     return idx_lb


# In[3]:


#### Main ####
# DATA_NAME = 'mimic319k48h'
# Settings for task, model, path, etc
# working_path = r'../..'

def get_time_series_sample_99plus_features_raw_Xhrs(args, hrs):
    HRS = hrs
    cachedir = Path(args.cachedir)
    working_path = cachedir.joinpath('admdata_99p', '{}hrs_raw'.format(HRS))
    # raw_data_path = os.path.join(working_path, 'data', DATA_NAME, 'raw')
    # processed_data_path = os.path.join(working_path, 'data', DATA_NAME)
    raw_data_path = working_path
    processed_data_path = os.path.join(working_path, 'series')
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    # LAB_EVENTS_IDX = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14]) # labevents and chartevents
    LAB_EVENTS_IDX = np.array(list(range(40, 73))+list(range(79, 102))+list(range(
        102, 110))+list(range(111, 117))+list(range(117, 129))+list(range(138, 140)))

    print('load data file')
    data_all = np.empty([0], dtype=object)
    for datanpz_file_name in ['DB_merged_%dhrs.npy' % HRS]:
        datanpz_file_pathname = os.path.join(raw_data_path,
                                             datanpz_file_name)
        data_all = np.concatenate(
            (data_all, np.load(datanpz_file_pathname, allow_pickle=True)))

    print('load icd9 label file')
    label_icd9_all = np.empty([0], dtype=object)
    for label_icd9_npz_file_name in ['ICD9-%dhrs.npy' % HRS]:
        label_icd9_npz_file_pathname = os.path.join(raw_data_path,
                                                    label_icd9_npz_file_name)
        label_icd9_all = np.concatenate((label_icd9_all,
                                         np.load(label_icd9_npz_file_pathname, allow_pickle=True)))

    # print('load icd9 subcat list file')
    # subcat_lbs = []
    # subcat_ubs = []
    # with open(os.path.join(raw_data_path, 'ICD9_subcat.csv'), 'r') as f:
    #     for line in f.readlines():
    #         subcat_id, subcat_lb, subcat_ub = line.split(',')
    #         subcat_lbs.append(subcat_lb)
    #         subcat_ubs.append(subcat_ub)
    #     subcat_lbs = np.array(subcat_lbs)
    #     subcat_ubs = np.array(subcat_ubs)

    print('load mor label file')
    label_mor_all = None
    for label_mor_npz_file_name in ['AGE_LOS_MORTALITY_%dhrs.npy' % HRS]:
        label_mor_npz_file_pathname = os.path.join(raw_data_path,
                                                   label_mor_npz_file_name)
        if label_mor_all is None:
            label_mor_all = np.load(
                label_mor_npz_file_pathname, allow_pickle=True)
        else:
            label_mor_all = np.concatenate((label_mor_all,
                                            np.load(label_mor_npz_file_pathname, allow_pickle=True)))

    print('load admission features')
    adm_features_all = np.load(os.path.join(
        raw_data_path, 'ADM_FEATURES_%dhrs.npy' % HRS), allow_pickle=True)

    print('load mortality labels')
    adm_labels_all = np.load(os.path.join(
        raw_data_path, 'ADM_LABELS_%dhrs.npy' % HRS), allow_pickle=True)

    N_all = len(data_all)
    print('# of samples:', N_all)
    # get per-frame samples;
    # imputed-normed-ep (imputation here):
    #               ep_tdata_raw, ep_tdata: N * [ti * D]
    #               ep_tdata_mean, ep_tdata_std: D
    # normed-ep:    X_t, X_t_mask, deltaT_t: N * [ti * D]
    #               T_t: N * [ti]
    X_raw_p48 = np.array([np.array(xx, dtype=float)[:, :-2]
                          for xx in data_all])
    tsraw_p48 = np.array([np.array(xx, dtype=float)[:, -2] for xx in data_all])
    del data_all

    idx_x = np.where([(tt[-1] - tt[0]) > 1.0*60*60*HRS for tt in tsraw_p48])[0]
    idx_x2 = np.where([(tt[-1] - tt[0]) <= 1.0*60 *
                       60*HRS for tt in tsraw_p48])[0]
    print(idx_x2)
    N = len(idx_x)
    print('# of samples > %s hours:' % (HRS), N)
    assert N_all == N
    X_raw = X_raw_p48[idx_x]
    tsraw = tsraw_p48[idx_x]
    label_icd9_all = label_icd9_all[idx_x]
    label_mor_all = label_mor_all[idx_x]
    adm_features_all = adm_features_all[idx_x]
    adm_labels_all = adm_labels_all[idx_x]

    for i_n in range(N):
        # print i_n
        if i_n % 20 == 0:
            print('.', end='')
            sys.stdout.flush()
        for i_t in range(len(X_raw[i_n])):
            for i_d in range(len(X_raw[i_n][i_t])):
                if X_raw[i_n][i_t][i_d] is None:
                    X_raw[i_n][i_t][i_d] = np.nan
    X_raw_all = np.concatenate(X_raw)
    print('done!')

    # In[4]:

    # remove the columns with less observations
    print('get mr and kept idx')
    val_mr = np.sum(np.isnan(X_raw_all), axis=0) * 1.0 / X_raw_all.shape[0]
    keep_val_idx = val_mr < 1-5e-4
    keep_val_idx_list = np.where(keep_val_idx)
    X_raw_all_kept = X_raw_all[:, keep_val_idx]
    X_raw_kept = np.array([xx[:, keep_val_idx] for xx in X_raw])
    lab_events_idx = LAB_EVENTS_IDX

    del X_raw_all
    del X_raw

    # X_raw_all_

    # In[13]:

    keep_val_idx_list

    # In[17]:

    old_map_feature_colids = np.load(cachedir.joinpath(
        'admdata_99p/raw', 'map_feature_colids.npy'), allow_pickle=True).tolist()
    map_feature_colids = {}
    for key, value in old_map_feature_colids.items():
        try:
            nvalue = np.where(keep_val_idx_list[0] == value)[0][0]
        except:
            nvalue = None
        if nvalue is not None:
            map_feature_colids[key] = nvalue
    map_feature_colids

    # In[18]:

    X_raw_kept[0].shape

    # ## Generate non-temporal features for time series
    #
    # Here we generate non-temporal features for time series for the process of SuperLearner method.
    #
    # Since it is hard to manually decide what stats to use for each feature:
    # - For all features, we use minimum, maximum and average.
    # - For urinary output, we also use sum.
    #
    # After this step, we get the file 'tsmean_Xhrs.npz' for generating input files for SuperLearner method.

    # In[19]:

    non_series_dir = os.path.join(processed_data_path, '../non_series')
    if not os.path.exists(non_series_dir):
        os.makedirs(non_series_dir)

    minmaxavg_list = list(range(X_raw_kept[0].shape[1]))
    sum_list = list(
        map(lambda x: map_feature_colids[x], ['urinary_output_sum']))
    # adm_features_all
    total_featuren = len(minmaxavg_list)*3 + len(sum_list) * \
        1 + adm_features_all.shape[1]
    hrs_mean_array = np.full((N, total_featuren), np.nan)
    for i in range(N):
        if i % 20 == 0:
            print('.', end='')
            sys.stdout.flush()
        tsraw[i] = tsraw[i].flatten()
        t = 0
        while t < len(tsraw[i]) and tsraw[i][t] - tsraw[i][0] <= HRS * 3600.0:
            t = t + 1
        fstart = 0
    #     # min_list
    #     tempmin = np.nanmin(X_raw_kept[i][0:t, min_list], axis=0)
    #     hrs_mean_array[i, fstart:fstart+len(min_list)*1] = tempmin
    #     fstart += len(min_list)*1
    #     # minmax_list
    #     tempmin = np.nanmin(X_raw_kept[i][0:t, minmax_list], axis=0)
    #     tempmax = np.nanmax(X_raw_kept[i][0:t, minmax_list], axis=0)
    #     hrs_mean_array[i, fstart:fstart+len(minmax_list)*2] = np.concatenate([tempmin, tempmax])
    #     fstart += len(minmax_list)*2
        # mimmaxavg_list
        tempmin = np.nanmin(X_raw_kept[i][0:t, minmaxavg_list], axis=0)
        tempmax = np.nanmax(X_raw_kept[i][0:t, minmaxavg_list], axis=0)
        tempavg = np.nanmean(X_raw_kept[i][0:t, minmaxavg_list], axis=0)
        hrs_mean_array[i, fstart:fstart +
                       len(minmaxavg_list)*3] = np.concatenate([tempmin, tempmax, tempavg])
        fstart += len(minmaxavg_list)*3
        # sum_list
        tempsum = np.nansum(X_raw_kept[i][0:t, sum_list], axis=0)
        hrs_mean_array[i, fstart:fstart+len(sum_list)*1] = tempsum
        fstart += len(sum_list)*1
        # static list
        hrs_mean_array[i, fstart:] = adm_features_all[i, :]

    hrs_mean_labels = adm_labels_all
    np.savez_compressed(os.path.join(non_series_dir, 'tsmean_%dhrs.npz' %
                                     HRS), hrs_mean_array=hrs_mean_array, hrs_mean_labels=hrs_mean_labels)

    # In[20]:

    print('get mean and std for tdata')
    # last frame is time t in seconds
    n_temporal_var = X_raw_all_kept.shape[1]
    ep_tdata_mean = np.nanmean(X_raw_all_kept, axis=0)
    ep_tdata_std = np.nanstd(X_raw_all_kept, axis=0)
    del X_raw_all_kept

    # get ep data with mask and deltaT
    # 0-mean, 1-std, merge observations within 5 mins
    merging_mins = 5
    print('get X_new and t_new')
    X_new = np.empty([N], dtype=object)
    t_new = np.empty([N], dtype=object)
    for i in range(N):
        if i % 20 == 0:
            print('.', end='')
            sys.stdout.flush()
        tsraw[i] = tsraw[i].flatten()
        t = 0
        X_new[i] = []
        t_new[i] = []
        while t < len(tsraw[i]):
            t1 = t+1
            while t1 < len(tsraw[i]) and tsraw[i][t1] - tsraw[i][t] <= merging_mins*60:
                t1 += 1
            # merge [t:t1]
    #         X_new[i].append(
    #             (np.nanmean(X_raw_kept[i][t:t1,:], axis=0) - ep_tdata_mean) \
    #                 /ep_tdata_std
    #             )
            # Here we do not normalize the data!!!
            X_new[i].append(
                np.nanmean(X_raw_kept[i][t:t1, :], axis=0)
            )
            # X_new[i].append(np.nanmean(X_raw_kept[i][t:t1,:], axis=0))
            t_new[i].append(int((tsraw[i][t1-1]+tsraw[i][t])/2))
            t = t1
    print('done!')

    # In[21]:

    print('get X_t, mask, etc')
    X_t = np.empty([N], dtype=object)        # N * [t*d]
    X_t_mask = np.empty([N], dtype=object)   # N * [t*d]
    T_t = t_new                                 # N * [t]
    deltaT_t = np.empty([N], dtype=object)   # N * [t*d]
    for i in range(N):
        if i % 20 == 0:
            print('.', end='')
            sys.stdout.flush()
        X_t[i] = np.vstack(X_new[i])
        X_t_mask[i] = 1-np.isnan(X_t[i]).astype('int8')
        X_t[i][np.isnan(X_t[i])] = 0
        deltaT_t[i] = np.zeros_like(X_t[i], dtype=int)
        deltaT_t[i][0, :] = 0
        for i_t in range(1, len(T_t[i])):
            deltaT_t[i][i_t, :] = T_t[i][i_t] - T_t[i][i_t-1] + \
                (1-X_t_mask[i][i_t-1, :]) * deltaT_t[i][i_t-1, :]
    print('done!')
    del X_new

    # In[22]:

    # extract subcat labels
    # for i_n, label_i in enumerate(label_icd9_all):
    #     for i_li, label_vec in enumerate(label_i):
    #         subcat = get_icd9_subcat_label(label_vct[2])
    #         label_i[i_li].append(subcat)
    #     label_icd9_all[i_n] = label_i

    # get labels
    print('get labels')
    class_icd9_counts = np.bincount(
        np.concatenate(label_icd9_all)[:, 3].astype(int))
    class_icd9_list = np.where(class_icd9_counts > 10)[0]
    class_icd9_list.sort()

    # class_icd9_subcat_counts = np.bincount(
    #     np.concatenate(label_icd9_all)[:,4].astype(int))
    # class_icd9_subcat_list = np.where(class_icd9_subcat_counts >= 200)[0]
    # class_icd9_subcat_list.sort()

    n_class_icd9 = class_icd9_list.shape[0]
    # n_class_icd9_subcat = class_icd9_subcat_list.shape[0]
    y_icd9 = np.zeros([N, n_class_icd9], dtype=int)
    # y_icd9_subcat = np.zeros([N, n_class_icd9_subcat], dtype=int)
    for i_n, label_i in enumerate(label_icd9_all):
        for label_vec in label_i:
            class_idx = np.array(
                [cl == label_vec[3] for cl in class_icd9_list],
                dtype=bool)
            y_icd9[i_n][class_idx] = 1
    #             subcat_idx = np.array(
    #                 [cl == label_vec[4] for cl in class_icd9_subcat_list],
    #                 dtype=bool)
    #             y_icd9_subcat[i_n][subcat_idx] = 1

    y_mor = np.expand_dims(np.array(label_mor_all[:, 4], dtype=int), axis=1)
    age_days = label_mor_all[:, 2]
    y_los = label_mor_all[:, 3]

    # print('# of class, subcat:', n_class_icd9, n_class_icd9_subcat)
    print('# of class, subcat:')

    np.savez_compressed(os.path.join(processed_data_path, 'normed-ep-stats.npz'),
                        class_icd9_list=class_icd9_list,
                        class_icd9_counts=class_icd9_counts,
                        #          class_icd9_subcat_list=class_icd9_subcat_list,
                        #          class_icd9_subcat_counts=class_icd9_subcat_counts,
                        keep_val_idx_list=keep_val_idx_list,
                        ep_tdata_mean=ep_tdata_mean, ep_tdata_std=ep_tdata_std,
                        n_class_icd9=n_class_icd9,
                        #          n_class_icd9_subcat=n_class_icd9_subcat,
                        N=N, val_mr=val_mr, idx_x=idx_x, age_days=age_days)

    np.savez_compressed(os.path.join(processed_data_path, 'normed-ep.npz'),
                        X_t=X_t, X_t_mask=X_t_mask, T_t=T_t, deltaT_t=deltaT_t,
                        y_icd9=y_icd9, y_mor=y_mor, adm_features_all=adm_features_all, adm_labels_all=adm_labels_all, y_los=y_los)
    # , y_icd9_subcat=y_icd9_subcat)

    del X_t, X_t_mask, deltaT_t

    # In[23]:

    # get first N hours data
    # one data sample for one patient
    # hours_list = [(2, 24), (1, 24), (1, 48), (2, 48)]
    hours_list = [(2, HRS), (1, HRS)]
    for n_sample_hour, n_full_hour in hours_list:
        print('get X_miss', n_sample_hour, n_full_hour)
        #n_sample_hour = 2
        #n_full_hour = HRS
        n_time_step = int(n_full_hour / n_sample_hour)
        # get X_miss first from X_raw_all_kept and tsraw, (sampled)
        X_miss = np.empty([N], dtype=object)
        T_miss = np.zeros([N], dtype=int)
        for i_n in range(N):
            if i_n % 20 == 0:
                print('.', end='')
                sys.stdout.flush()
            T_miss[i_n] = math.ceil(
                (tsraw[i_n][-1]-tsraw[i_n][0])*1.0/(60*60*n_sample_hour))
            X_miss[i_n] = np.zeros([T_miss[i_n], n_temporal_var], dtype=float)
            for i_t in range(T_miss[i_n]):
                t_idx = np.logical_and(
                    (tsraw[i_n]-tsraw[i_n][0]) >= i_t*(60*60*n_sample_hour),
                    (tsraw[i_n]-tsraw[i_n][0]) <= (1+i_t) * (60*60*n_sample_hour))
                X_raw_thist = X_raw_kept[i_n][t_idx, :]
                # Here we do not normalize the data!!!
    #             X_miss[i_n][i_t,:] = \
    #                 (np.nanmean(X_raw_thist, axis=0) - ep_tdata_mean) / ep_tdata_std
                X_miss[i_n][i_t, :] = np.nanmean(X_raw_thist, axis=0)
        print('done!')
        # X_imputed: do forward/backward imputing from X_miss for lab events
        #            do mean imputing for other events
        print('get X_imputed')
        X_imputed = deepcopy(X_miss)
        for i_n in range(N):
            if i_n % 20 == 0:
                print('.', end='')
                sys.stdout.flush()
            i_n_mean = np.nanmean(X_imputed[i_n], axis=0)
            for i_t in range(1, T_miss[i_n]):
                for i_d in range(n_temporal_var):
                    if np.isnan(X_imputed[i_n][i_t, i_d]):
                        if keep_val_idx_list[0][i_d] in lab_events_idx:
                            X_imputed[i_n][i_t,
                                           i_d] = X_imputed[i_n][i_t-1, i_d]
            for i_t in range(T_miss[i_n]-2, -1, -1):
                for i_d in range(n_temporal_var):
                    if np.isnan(X_imputed[i_n][i_t, i_d]):
                        if keep_val_idx_list[0][i_d] in lab_events_idx:
                            X_imputed[i_n][i_t,
                                           i_d] = X_imputed[i_n][i_t+1, i_d]
            # X_imputed[i_n][np.isnan(X_imputed[i_n])] = 0
            # Here we use mean value of each feature in current time series to impute nans
            for i_t in range(0, T_miss[i_n]):
                for i_d in range(n_temporal_var):
                    if np.isnan(X_imputed[i_n][i_t, i_d]):
                        X_imputed[i_n][i_t, i_d] = i_n_mean[i_d]
            # for values which are still none, just impute with 0
    #         X_imputed[i_n][np.isnan(X_imputed[i_n])] = 0
        print('done!')

        # get first # hours, for both data and masking
        print('get ep_tdata')
        ep_tdata = np.zeros([N, n_time_step, n_temporal_var], dtype=float)
        ep_tdata_masking = np.zeros_like(ep_tdata, dtype=int)
        for i_n in range(N):
            if i_n % 20 == 0:
                print('.', end='')
                sys.stdout.flush()
            xx_imp = X_imputed[i_n]
            xx_mis = X_miss[i_n]
            tt_min = min(n_time_step, len(xx_imp))
            assert tt_min > 0
            ep_tdata[i_n, :tt_min, :] = xx_imp[:tt_min, :]
            ep_tdata[i_n, tt_min:, :] = ep_tdata[i_n, tt_min-1, :][None, :]
            ep_tdata_masking[i_n, :tt_min, :] = (
                ~np.isnan(xx_mis[:tt_min, :])).astype(int)
        print('done!')

        ep_data = np.reshape(ep_tdata, [N, n_time_step*n_temporal_var])
        ep_data_masking = np.reshape(
            ep_tdata_masking, [N, n_time_step*n_temporal_var])

        np.savez_compressed(os.path.join(processed_data_path,
                                         'imputed-normed-ep' + '_' + str(n_sample_hour) +
                                         '_' + str(n_full_hour) + '.npz'),
                            ep_data=ep_data, ep_tdata=ep_tdata,
                            ep_data_masking=ep_data_masking,
                            ep_tdata_masking=ep_tdata_masking,
                            y_icd9=y_icd9, y_mor=y_mor, adm_features_all=adm_features_all, adm_labels_all=adm_labels_all, y_los=y_los)
    #     , y_icd9_subcat=y_icd9_subcat)

    # In[24]:

    print(np.mean(y_mor))

    # In[25]:

    # imputed_data = np.load('../../Data/admdata_17f/24hrs_raw/series/imputed-normed-ep_1_24.npz')
    # y_icd9 = imputed_data['y_icd9']
    # adm_labels_all = imputed_data['adm_labels_all']

    print('make splits')
    # make 5-fold cv splits if file not exists

    def make_splits_on(y_mor, foldn):
        folds_ep_mor = []
        for i in range(1):
            folds_ep_mor.append(make_splits(y_mor, foldn))
        return folds_ep_mor

    def gen_folds_ids(foldn, fold_file_path, **kwargs):
        # generate folds based on label sets
        folds = {}
        print(list(kwargs.items()))
        for labelname, (labelarray, is_multi_task) in kwargs.items():
            assert len(labelarray.shape) > 1
            folds[labelname] = []
            if is_multi_task:
                for ln in range(labelarray.shape[1]):
                    tempy = labelarray[:, ln]
                    try:
                        lnfold = make_splits_on(tempy, foldn)
                    except:
                        print('pass {0} {1}'.format(labelname, ln))
                        lnfold = None
                    folds[labelname].append(lnfold)
            else:
                folds[labelname].append(make_splits_on(labelarray, foldn))
        np.savez_compressed(fold_file_path, **folds)
        return folds

    def get_standardize_stats_for_training(ep_tdata, ep_tdata_masking, adm_features_all, training_ids):
        trainset = ep_tdata[training_ids]
        trainset_masking = ep_tdata_masking[training_ids]
        train_admfeatures = adm_features_all[training_ids]
        id_num = trainset.shape[0]
        dim = trainset.shape[2]
        stats = np.empty((dim, 2)) * np.nan
        for d in range(dim):
            dim_values = trainset[:, :, d].flatten()
            dim_mean = np.nanmean(dim_values)
            dim_std = np.nanstd(dim_values)
            stats[d, :] = np.array([dim_mean, dim_std])
        nsdim = adm_features_all.shape[1]
        nsstats = np.empty((nsdim, 2)) * np.nan
        for d in range(nsdim):
            dim_values = train_admfeatures[:, d].flatten()
            dim_mean = np.nanmean(dim_values)
            dim_std = np.nanstd(dim_values)
            nsstats[d, :] = np.array([dim_mean, dim_std])
        return stats, nsstats

    def get_standardize_stats_for_training_missing(ep_tdata, ep_tdata_masking, adm_features_all, training_ids):
        trainset = np.concatenate(ep_tdata[training_ids])
        trainset_masking = np.concatenate(ep_tdata_masking[training_ids])
        train_admfeatures = adm_features_all[training_ids]
        id_num = trainset.shape[0]
        dim = trainset.shape[1]
        stats = np.empty((dim, 2)) * np.nan
        for d in range(dim):
            dim_masking = trainset_masking[:, d].flatten()
            dim_values = trainset[:, d].flatten()[np.where(dim_masking == 1)]
            dim_mean = np.nanmean(dim_values)
            dim_std = np.nanstd(dim_values)
            stats[d, :] = np.array([dim_mean, dim_std])
        nsdim = adm_features_all.shape[1]
        nsstats = np.empty((nsdim, 2)) * np.nan
        for d in range(nsdim):
            dim_values = train_admfeatures[:, d].flatten()
            dim_mean = np.nanmean(dim_values)
            dim_std = np.nanstd(dim_values)
            nsstats[d, :] = np.array([dim_mean, dim_std])
        return stats, nsstats

    def get_standardize_stats_for_folds(folds, stdfunc, ep_tdata, ep_tdata_masking, adm_features_all):
        statsdict = {}
        for key, value in folds.items():
            statsdict[key] = []
            for folds_ids in value:
                foldsstat = []
                for folds_ep_mor in folds_ids:
                    foldsn = folds_ep_mor.shape[0]
                    stats = []
                    ep_tdata_stdized_list = []
                    for foldn in range(foldsn):
                        training_ids = folds_ep_mor[foldn, 0]
                        stat, nsstat = stdfunc(ep_tdata=ep_tdata, ep_tdata_masking=ep_tdata_masking,
                                               adm_features_all=adm_features_all, training_ids=training_ids)
                        fstat = [stat[:, 0], stat[:, 1]]
                        fnsstat = [nsstat[:, 0], nsstat[:, 1]]
                        stats.append([fstat, fnsstat])
                    foldsstat.append(np.array(stats))
                statsdict[key].append(foldsstat)
        return statsdict

    def split_dataset(datasetfilename, ep_tdata_attr, ep_tdata_masking_attr, ep_adm_features_all_attr, aidwhere, statfunc, foldn, fold_filedir, **kwargs):
        dataset = np.load(os.path.join(processed_data_path,
                                       datasetfilename + '.npz'), allow_pickle=True)
        subdataset = {}
        for key, value in dataset.items():
            subdataset[key] = value[aidwhere]
        sub_tdata = subdataset[ep_tdata_attr]
        sub_masking = subdataset[ep_tdata_masking_attr]
        sub_label_all = subdataset[ep_adm_features_all_attr]
        sublabelset = {}
        for key, (value, is_multi_task) in kwargs.items():
            sublabelset[key] = (value[aidwhere], is_multi_task)
        if not os.path.exists(fold_filedir):
            os.makedirs(fold_filedir)
        fold_file_path = os.path.join(fold_filedir, '%d-folds.npz' % foldn)
        folds = gen_folds_ids(
            foldn=foldn, fold_file_path=fold_file_path, **sublabelset)
        statsdict = get_standardize_stats_for_folds(
            folds, statfunc, ep_tdata=sub_tdata, ep_tdata_masking=sub_masking, adm_features_all=sub_label_all)
        np.savez_compressed(os.path.join(
            fold_filedir, datasetfilename+'-stdized.npz'), **statsdict)
        if not os.path.exists(os.path.join(fold_filedir, datasetfilename+'.npz')):
            np.savez_compressed(os.path.join(
                fold_filedir, datasetfilename+'.npz'), **subdataset)
        print('finish', fold_filedir)

    # select ids in carevue
    sql = 'select distinct hadm_id from mimiciii.icustays where dbsource = \'metavision\' '
    sql += 'UNION select distinct hadm_id from mimiciii.transfers where dbsource = \'metavision\''
    conn = getConnection()
    cur = conn.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    mvaids = sorted([r[0] for r in res])
    mvaidset = set(mvaids)

    MVDIR = os.path.join(processed_data_path, 'mv')
    CVDIR = os.path.join(processed_data_path, 'cv')
    ALLDIR = processed_data_path
    data_all = np.load(os.path.join(
        working_path, 'DB_merged_%dhrs.npy' % HRS), allow_pickle=True)
    allaids = np.array([t[0][-1] for t in data_all])
    mvwhere = np.array([aid in mvaidset for aid in allaids])
    cvwhere = ~mvwhere
    allwhere = np.logical_or(mvwhere, cvwhere)
    assert np.alltrue(allwhere)

    file_list = ['imputed-normed-ep_1_%d' %
                 HRS, 'imputed-normed-ep_2_%d' % HRS]
    for filename in file_list:
        for ids, dirname in zip([mvwhere, cvwhere, allwhere], [MVDIR, CVDIR, ALLDIR]):
            split_dataset(
                datasetfilename=filename,
                ep_tdata_attr='ep_tdata',
                ep_tdata_masking_attr='ep_tdata_masking',
                ep_adm_features_all_attr='adm_features_all',
                aidwhere=ids,
                statfunc=get_standardize_stats_for_training,
                foldn=5,
                fold_filedir=dirname,
                folds_ep_icd9=(y_icd9, True),
                folds_ep_icd9_multi=(y_icd9, False),
                folds_ep_mor=(adm_labels_all, True)
            )

    ep_datafilename = 'normed-ep'
    for ids, dirname in zip([mvwhere, cvwhere, allwhere], [MVDIR, CVDIR, ALLDIR]):
        split_dataset(
            datasetfilename=ep_datafilename,
            ep_tdata_attr='X_t',
            ep_tdata_masking_attr='X_t_mask',
            ep_adm_features_all_attr='adm_features_all',
            aidwhere=ids,
            statfunc=get_standardize_stats_for_training_missing,
            foldn=5,
            fold_filedir=dirname,
            folds_ep_icd9=(y_icd9, True),
            folds_ep_icd9_multi=(y_icd9, False),
            folds_ep_mor=(adm_labels_all, True)
        )


def get_time_series_sample_99plus_features_raw(args):
    get_time_series_sample_99plus_features_raw_Xhrs(args, 24)
    get_time_series_sample_99plus_features_raw_Xhrs(args, 48)
