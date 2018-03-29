
import argparse
import copy
import cPickle
import os
import sys
import warnings

import numpy as np
import sklearn
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
import theano

from tengwar.nnet.classifiers import CombinedLSTMandFFN, FeedForwardNetwork, SimpleLSTMNetwork
from util.gbt import Monitor


# To omit sklearn (deprecated) warning.
warnings.filterwarnings('ignore')

# standardizer for folds
class FoldsStandardizer(object):
    def __init__(self, serial_series, non_serial_series):
        self.serial_mean = serial_series[0]
        self.serial_std = serial_series[1]
        self.non_serial_mean = non_serial_series[0]
        self.non_serial_std = non_serial_series[1]

    def transform(self, X):
        print(X[0].shape)
        print(X[1].shape)
        assert len(X) == 2
        assert len(X[1].shape) == 3 # (id, time, feature)
        assert len(X[0].shape) == 2 # (id, feature)
        assert X[1].shape[2] == self.serial_mean.shape[0]
        assert X[1].shape[2] == self.serial_std.shape[0]
        assert X[0].shape[1] == self.non_serial_mean.shape[0]
        assert X[0].shape[1] == self.non_serial_std.shape[0]
        non_serial = np.copy(X[0])
        for id in xrange(non_serial.shape[0]):
            non_serial[id, :] = (non_serial[id, :] - self.non_serial_mean) / self.non_serial_std
        non_serial[np.isinf(non_serial)] = 0
        non_serial[np.isnan(non_serial)] = 0
        serial = np.copy(X[1])
        for id in xrange(serial.shape[0]):
            for t in xrange(serial.shape[1]):
                serial[id, t, :] = (serial[id, t, :] - self.serial_mean) / self.serial_std
        serial[np.isinf(serial)] = 0
        serial[np.isnan(serial)] = 0
        return [non_serial, serial]

class StaticFeaturesStandardizer(object):
    def __init__(self, train_mean, train_std):
        self.train_mean = train_mean
        self.train_std = train_std

    def transform(self, X):
        Xtrans = (X - self.train_mean) / self.train_std
        Xtrans[np.isinf(Xtrans)] = 0.0
        Xtrans[np.isnan(Xtrans)] = 0.0
        return Xtrans

class SAPSIITransformer(object):
    def __init__(self, train_idx):
        self.train_idx = train_idx

    def transform(self, X):
        '''
        [('GCS', 0)], 'mengcz_vital_ts': [('SysBP_Mean', 1), ('HeartRate_Mean', 2), ('TempC_Mean', 3)],
        'mengcz_pao2fio2_ts': [('PO2', 4), ('FIO2', 5)], 'mengcz_urine_output_ts': [('UrineOutput', 6)],
        'mengcz_labs_ts': [('BUN_min', 7), ('WBC_min', 8), ('BICARBONATE_min', 9), ('SODIUM_min', 10),
        ('POTASSIUM_min', 11), ('BILIRUBIN_min', 12)]

        age: 0, aids: 1, he,: 2, mets: 3, admissiontype: 4
        '''
        non_serial = np.copy(X[0])
        serial = np.copy(X[1])

        for admid in range(non_serial.shape[0]):
            # non_serial
            age, aids, hem, mets, admissiontype = non_serial[admid][0], non_serial[admid][1], non_serial[admid][2], non_serial[admid][3], non_serial[admid][4]

            try:
                age = age / 365.25
                if age < 40:
                    non_serial[admid][0] = 0.0
                elif age < 60:
                    non_serial[admid][0] = 7.0
                elif age < 70:
                    non_serial[admid][0] = 12.0
                elif age < 75:
                    non_serial[admid][0] = 15.0
                elif age < 80:
                    non_serial[admid][0] = 16.0
                elif age >= 80:
                    non_serial[admid][0] = 18.0
            except:
                non_serial[0] = 0.0

            try:
                if aids == 1:
                    non_serial[admid][1] = 17.0
                else:
                    non_serial[admid][1] = 0.0
            except:
                non_serial[admid][1] = 0.0

            try:
                if hem == 1:
                    non_serial[admid][2] = 10.0
                else:
                    non_serial[admid][2] = 0.0
            except:
                non_serial[admid][2] = 0.0

            try:
                if mets == 1:
                    non_serial[admid][3] = 9.0
                else:
                    non_serial[admid][3] = 0.0
            except:
                non_serial[admid][3] = 0.0

            try:
                if admissiontype == 0: # medical
                    non_serial[admid][4] = 6.0
                elif admissiontype == 1: # sche
                    non_serial[admid][4] = 0.0
                elif admissiontype == 2: # unsche
                    non_serial[admid][4] = 8.0
            except:
                non_serial[admid][4] = 0.0

            # serial
            for t in range(serial[admid].shape[0]):
                gcs = serial[admid][t][0]
                sbp = serial[admid][t][1]
                hr = serial[admid][t][2]
                bt = serial[admid][t][3]
                pfr = serial[admid][t][4]
                uo = serial[admid][t][5]
                sunl = serial[admid][t][6]
                wbc = serial[admid][t][7]
                sbl = serial[admid][t][8]
                sl = serial[admid][t][9]
                pl = serial[admid][t][10]
                bl = serial[admid][t][11]

                try:
                    if hr < 40:
                        serial[admid][t][2] = 11.0
                    elif hr >= 160:
                        serial[admid][t][2] = 7.0
                    elif hr >= 120:
                        serial[admid][t][2] = 4.0
                    elif hr < 70:
                        serial[admid][t][2] = 2.0
                    elif hr >= 70 and hr < 120:
                        serial[admid][t][2] = 0.0
                    else:
                        serial[admid][t][2] = 0.0
                except:
                    serial[admid][t][2] = 0.0

                try:
                    if sbp < 70:
                        serial[admid][t][1] = 13.0
                    elif sbp < 100:
                        serial[admid][t][1] = 5.0
                    elif sbp >= 200:
                        serial[admid][t][1] = 2.0
                    elif sbp >= 100 and sbp < 200:
                        serial[admid][t][1] = 0.0
                    else:
                        serial[admid][t][1] = 0.0
                except:
                    serial[admid][t][1] = 0.0

                try:
                    if bt < 39.0:
                        serial[admid][t][3] = 0.0
                    elif bt >= 39.0:
                        serial[admid][t][3] = 3.0
                    else:
                        serial[admid][t][3] = 0.0
                except:
                    serial[admid][t][3] = 0.0

                try:
                    if pfr < 100:
                        serial[admid][t][4] = 11.0
                    elif pfr < 200:
                        serial[admid][t][4] = 9.0
                    elif pfr >= 200:
                        serial[admid][t][4] = 6.0
                    else:
                        serial[admid][t][4] = 0.0
                except:
                    serial[admid][t][4] = 0.0

                try:
                    if uo < 500:
                        serial[admid][t][5] = 11.0
                    elif uo < 1000:
                        serial[admid][t][5] = 4.0
                    elif uo >= 1000:
                        serial[admid][t][5] = 0.0
                    else:
                        serial[admid][t][5] = 0.0
                except:
                    serial[admid][t][5] = 0.0

                try:
                    if sunl < 28.0:
                        serial[admid][t][6] = 0.0
                    elif sunl < 83.0:
                        serial[admid][t][6] = 6.0
                    elif sunl >= 84.0:
                        serial[admid][t][6] = 10.0
                    else:
                        serial[admid][t][6] = 0.0
                except:
                    serial[admid][t][6] = 0.0

                try:
                    if wbc < 1.0:
                        serial[admid][t][7] = 12.0
                    elif wbc >= 20.0:
                        serial[admid][t][7] = 3.0
                    elif wbc >= 1.0 and wbc < 20.0:
                        serial[admid][t][7] = 0.0
                    else:
                        serial[admid][t][7] = 0.0
                except:
                    serial[admid][t][7] = 0.0

                try:
                    if pl < 3.0:
                        serial[admid][t][10] = 3.0
                    elif pl >= 5.0:
                        serial[admid][t][10] = 3.0
                    elif pl >= 3.0 and pl < 5.0:
                        serial[admid][t][10] = 0.0
                    else:
                        serial[admid][t][10] = 0.0
                except:
                    serial[admid][t][10] = 0.0

                try:
                    if sl < 125:
                        serial[admid][t][9] = 5.0
                    elif sl >= 145:
                        serial[admid][t][9] = 1.0
                    elif sl >= 125 and sl < 145:
                        serial[admid][t][9] = 0.0
                    else:
                        serial[admid][t][9] = 0.0
                except:
                    serial[admid][t][9] = 0.0

                try:
                    if sbl < 15.0:
                        serial[admid][t][8] = 5.0
                    elif sbl < 20.0:
                        serial[admid][t][8] = 3.0
                    elif sbl >= 20.0:
                        serial[admid][t][8] = 0.0
                    else:
                        serial[admid][t][8] = 0.0
                except:
                    serial[admid][t][8] = 0.0

                try:
                    if bl < 4.0:
                        serial[admid][t][11] = 0.0
                    elif bl < 6.0:
                        serial[admid][t][11] = 4.0
                    elif bl >= 6.0:
                        serial[admid][t][11] = 9.0
                    else:
                        serial[admid][t][11] = 0.0
                except:
                    serial[admid][t][11] = 0.0

                try:
                    if gcs < 3:
                        serial[admid][t][0] = 0.0
                    elif gcs < 6:
                        serial[admid][t][0] = 26.0
                    elif gcs < 9:
                        serial[admid][t][0] = 13.0
                    elif gcs < 11:
                        serial[admid][t][0] = 7.0
                    elif gcs < 14:
                        serial[admid][t][0] = 5.0
                    elif gcs >= 14 and gcs <= 15:
                        serial[admid][t][0] = 0.0
                    else:
                        serial[admid][t][0] = 0.0
                except:
                    serial[admid][t][0] = 0.0
        non_serial_mean, non_serial_std = np.nanmean(non_serial[self.train_idx], axis=0), np.nanstd(non_serial[self.train_idx], axis=0)
        non_serial = (non_serial - non_serial_mean) / non_serial_std
        non_serial[np.isnan(non_serial)] = 0.0
        non_serial[np.isinf(non_serial)] = 0.0

        serial_mean, serial_std = np.nanmean(np.concatenate(serial[self.train_idx], axis=0), axis=0), np.nanstd(np.concatenate(serial[self.train_idx], axis=0), axis=0)
        serial = (serial - serial_mean) / serial_std
        serial[np.isnan(serial)] = 0.0
        serial[np.isinf(serial)] = 0.0

        return [non_serial, serial]

def list_with_index(X, idx=None):
    if type(X) is np.ndarray:
        if idx is None:
            return X
        else:
            return X[idx]
    return [list_with_index(x, idx) for x in X]


# Learner Methods
def train_and_test_learner(
        X,  # (n_samples, ...) array
            # (n_samples, n_features) for static/flattened input
            # (n_samples, n_timesteps, n_dimensions) for temporal input
            # [(X1, X2, etc)] list of array for multi-modality input
        y,  # (n_samples, 1) array of labels (0/1 for binary labels)
        attrs,
        X_extra = None, y_extra = None,     # optional
            #  y_extra: (n_reps, n_folds, ...) array. Soft labels to train.
        clf = None,     # classifier
        tsf = None,     # transformer (used before classifier, optional)
        taskName = 'task_name', clfName = 'clf_name', 
        splits = None,  # (n_reps, n_folds, 2/3) array of list 
                        # (train, valid, test) or (train, test) splits
        copied_X = False,   # True: X is (n_reps, n_folds, ...)
                            # False: X is (...)
        skip_trained = False,    # True: skip training if files exist.
        ):
    '''
    Detail:
    X_splitted, X_extra_splitted: (n_samples, n_features) array
    X_input: Can be
        1) X_splitted; 
        2) tsf(X_splitted); 
        3) (X_splitted, tsf(X_extra_splitted))
    y_fitting: Target to train the model. Can be
        1) y_extra[rep][i_fold];
        2) y
    
    Output:
    Average score of accuracy, f1, precision, recall, AUC
    
    Return:
    List of array of (y_soft, y_pred, X_trans, feature_imp, clf)

    '''

    print 'start working on', taskName, 'with', clfName
    sys.stdout.flush()
    result_file_pathname = os.path.join(result_path, '_'.join([taskName, clfName])+'.npz')
    all_results_file_pathname = os.path.join(result_path, 'results-all.txt')
    model_file_pathname = os.path.join(model_path, '_'.join([taskName, clfName])+'.pkl')
    weights_file_pathname_pattern = os.path.join(model_path, '_'.join([taskName, clfName, 'rep{0}', 'fold{1}'])+'.h5')
    if skip_trained:
        if os.path.exists(result_file_pathname) and os.path.exists(model_file_pathname):
            try:
                print '\tSkip working on', taskName, 'with', clfName, 'since files exist...'
                with open(model_file_pathname, 'rb') as f:
                    clf_arr = cPickle.load(f)
                results = np.load(result_file_pathname)
                return (results['y_soft_arr'], results['y_pred_arr'], 
                        results['X_trans_arr'], results['feature_imp_arr'], clf_arr
                        )
            except:
                print 'Skip failed, redo the training...'
    n_reps = min(len(splits), 5)

    # Output variables
    cv_acc_scores = []
    cv_f1_scores = []
    cv_prec_scores = []
    cv_rec_scores = []
    cv_auc_scores = []
    cv_prc_scores = []
    cv_mse_scores = []
    
    # Return variables
    y_soft_arr = np.empty([n_reps], dtype=object)
    y_pred_arr = np.empty([n_reps], dtype=object)
    X_trans_arr = np.empty([n_reps], dtype=object)
    feature_imp_arr = np.empty([n_reps], dtype=object)
    clf_arr = np.empty([n_reps], dtype=object)
    
    # n repeats
    for rep in xrange(n_reps):
        y_soft_arr[rep] = []
        y_pred_arr[rep] = []
        X_trans_arr[rep] = []
        feature_imp_arr[rep] = []
        clf_arr[rep] = []

        fold_idxs = splits[rep]
        # n folds in each repeat
        for i_fold, fold_idx in enumerate(fold_idxs):
            # train/validation/test set or train/test set
            if len(fold_idx) == 2:
                idx_trva, idx_te = fold_idx
            elif len(fold_idx) == 3:
                idx_tr, idx_va, idx_te = fold_idx
                idx_trva =  np.concatenate((idx_tr, idx_va))
            # X is splitted for reps and folds or not.
            if copied_X:
                X_splitted = X[rep][i_fold]
                if X_extra is not None:
                    X_extra_splitted = X_extra[rep][i_fold]
                else:
                    X_extra_splitted = None
            else:
                X_splitted = X
                X_extra_splitted = X_extra
            y_fitting = y if y_extra is None else y_extra[rep][i_fold]
            
            # See if tsf is needed    
            if tsf is not None:
                if X_extra_splitted is None:
                    # use tsf on the whole X
                    if hasattr(tsf, '__iter__'):
                        X_input = tsf[rep][i_fold].transform(X_splitted)
                    else:
                        X_input = tsf.fit(
                                list_with_index(X_splitted, idx_trva), 
                                y_fitting[idx_trva]
                            ).transform(X_splitted)
                else:
                    # use tsf on X_extra and combine with X
                    if hasattr(tsf, '__iter__'):
                        X_input = np.hstack((
                            X_splitted, 
                            tsf[rep][i_fold].transform(X_extra_splitted)
                        ))
                    else:
                        X_input = np.hstack((
                            X_splitted, 
                            tsf.fit(
                                list_with_index(X_extra_splitted, idx_trva),
                                y_fitting[idx_trva]
                            ).transform(X_extra_splitted)
                        ))
            else:
                # No transform at all
                X_input = X_splitted
            # Train the model
            if isinstance(clf, sklearn.ensemble.BaseEnsemble) and len(fold_idx) == 3:
                # eary stopping for tree methods.
                clf = clf.fit(list_with_index(X_input, idx_tr), 
                              0.5*y_fitting[idx_tr].flatten() + 0.5*y[idx_tr].flatten(),
                              monitor = Monitor(list_with_index(X_input, idx_va),
                                                y[idx_va].flatten()))
           else: 
                #print  len(X_input), X_input[0].shape, X_input[1].shape  
                clf = clf.fit(list_with_index(X_input, idx_trva), y_fitting[idx_trva])
            # Handle the dtype bug in sklearn
            if isinstance(clf, GradientBoostingRegressor):
                ysoft = clf.decision_function(sklearn.utils.check_array(X_input, dtype = sklearn.tree._tree.DTYPE, order="C"))
            else:                    
                ysoft = clf.decision_function(X_input) if hasattr(clf, 'decision_function') else clf.predict(X_input)
            if taskName != 'Length_Of_Stay':
                ypred = 1*(clf.predict(X_input)>0)
            else:
                ypred = clf.predict(X_input)
            if isinstance(clf, sklearn.ensemble.BaseEnsemble):
                # tree methods
                Xtrans = np.zeros([np.shape(X_input)[0], np.shape(clf.estimators_)[0]])
                for i_est, ests in enumerate(clf.estimators_):
                    Xtrans[:, i_est] = ests[0].predict(X_input)
            else:            
                # other methods
                Xtrans = clf.transform(list_with_index(X_input))
            
            if hasattr(clf, 'feature_importances_'):
                feature_imp = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                feature_imp = abs(clf.coef_)
            else:
                feature_imp = None
                
            y_soft_arr[rep].append(ysoft)
            y_pred_arr[rep].append(ypred)
            X_trans_arr[rep].append(Xtrans)
            feature_imp_arr[rep].append(feature_imp)
            # Still need to fix this (error when do the cpickle dump)
            if (isinstance(clf, SimpleLSTMNetwork) 
                or isinstance(clf, CombinedLSTMandFFN)
                or isinstance(clf, FeedForwardNetwork)
                or isinstance(clf, HierarchicalMultimodal)
                ):
                # use (to avoid unnecessary model compiling)
                clf_arr[rep].append(clf.model.to_json())
                weights_file_pathname = \
                    weights_file_pathname_pattern.format(str(rep), str(i_fold))
                clf.model.save_weights(weights_file_pathname,overwrite=True)
            else:   
                # use deep copy and dump for other model.
                clf_arr[rep].append(copy.deepcopy(clf))
            if taskName != 'Length_Of_Stay':
                cv_acc_scores.append(metrics.accuracy_score(y[idx_te], ypred[idx_te]))
                cv_prec_scores.append(metrics.precision_score(y[idx_te], ypred[idx_te]))
                cv_rec_scores.append(metrics.recall_score(y[idx_te], ypred[idx_te]))
                cv_f1_scores.append(metrics.f1_score(y[idx_te], ypred[idx_te]))
                cv_auc_scores.append(metrics.roc_auc_score(y[idx_te], ysoft[idx_te]))
                cv_prc_scores.append(metrics.average_precision_score(y[idx_te], ysoft[idx_te]))
                print cv_auc_scores
                print cv_prc_scores
            else:
                cv_mse_scores.append(metrics.mean_squared_error(y[idx_te], ypred[idx_te]))
                print cv_mse_scores


    if taskName != 'Length_Of_Stay':
        metric_list = (cv_acc_scores,cv_f1_scores,cv_auc_scores,cv_prec_scores,cv_rec_scores,cv_prc_scores)
    else:
        metric_list = (cv_mse_scores,)
    print ('%s\t\t\t%s' %(taskName, clfName)),
    for metric in metric_list:
        print ('\t%.5f\t%.5f' % (np.mean(metric), np.std(metric))),
    with open(all_results_file_pathname, 'a') as f:
        f.write('%s\t\t\t%s' %(taskName, clfName))
        for metric in metric_list:
            f.write('\t%.5f\t%.5f' % (np.mean(metric), np.std(metric)))
        f.write('\n')
    np.savez(result_file_pathname,
             y_soft_arr=y_soft_arr, y_pred_arr=y_pred_arr,
             X_trans_arr=X_trans_arr, feature_imp_arr=feature_imp_arr)
    with open(model_file_pathname, 'wb') as f:
        cPickle.dump(clf_arr, f);
    return (y_soft_arr, y_pred_arr, X_trans_arr, feature_imp_arr, clf_arr)


def test_basic_learner(
        X, y, attrs, 
        clf = None,
        taskName = 'task', clfName = 'clf', tsf=None,
        splits = None):
    return train_and_test_learner(
            X=X, y=y, attrs=attrs, 
            X_extra = None, y_extra = None, 
            clf = clf, tsf = tsf,
            taskName = taskName, clfName = clfName, 
            copied_X = False, splits = splits)

def test_stacked_learner(
        X_splits, y,
        clf = None,
        taskName = 'task', clfName = 'clf', 
        splits = None):
    return train_and_test_learner(
            X=X_splits, y=y, attrs=None, 
            X_extra = None, y_extra = None,
            clf = clf, tsf = None,
            taskName = taskName, clfName = clfName, 
            copied_X = True, splits = splits)

def test_mimic_learner(
        X, y, y_soft_splits,
        clf = None, 
        taskName = 'task', clfName = 'clf', 
        splits = None):
    return train_and_test_learner(
            X=X, y=y, attrs=None, 
            X_extra = None, y_extra = y_soft_splits, 
            clf = clf, tsf = None,
            taskName = taskName, clfName = clfName, 
            copied_X = False, splits = splits)


#### Start here for the common configs ####
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('data_name', type=unicode)
arg_parser.add_argument('task_name', type=unicode)
arg_parser.add_argument('model_type', type=unicode, default= 1)
arg_parser.add_argument('data_file_name', type=unicode)
arg_parser.add_argument('folds_file_name', type=unicode)
arg_parser.add_argument('folds_stat_file_name', type=unicode)
arg_parser.add_argument('--static_features_path', type=unicode, default='')
arg_parser.add_argument('--label_type', type=int, default=0)
arg_parser.add_argument('--working_path', '-p', type=unicode, default='..')
# training
arg_parser.add_argument('--ffn_depth', type=int, default=4)
arg_parser.add_argument('--merge_depth', type=int, default=0)
arg_parser.add_argument('--output_dim', type=int, default=4)
arg_parser.add_argument('--batch_size', type=int, default=20)
arg_parser.add_argument('--nb_epoch', type=int, default=50)
arg_parser.add_argument('--early_stopping', type=str, default='True_BestWeight')
arg_parser.add_argument('--early_stopping_patience', type=int, default=10)
arg_parser.add_argument('--batch_normalization', type=str, default='False')
arg_parser.add_argument('--learning_rate', type=float, default=0.001)
arg_parser.add_argument('--dropout', type=float, default=0.1)
arg_parser.add_argument('--without_static', action='store_true')
arg_parser.add_argument('--remove_sapsii', action='store_true')
arg_parser.add_argument('--static_ffn_depth', type=int, default=2)
arg_parser.add_argument('--static_hidden_dim', type=int, default=None)
arg_parser.add_argument('--use_sapsii_scores', action='store_true')
try:
    args = arg_parser.parse_args()
    DATA_NAME = args.data_name
    TASK_NAME = args.task_name
    model_type = int(args.model_type)
    working_path = args.working_path
    data_file_name = args.data_file_name
    folds_file_name = args.folds_file_name
    folds_stat_file_name = args.folds_stat_file_name
    static_features_path = args.static_features_path
    label_type = args.label_type
    fit_parameters = [args.output_dim, args.ffn_depth, args.merge_depth]
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    early_stopping = args.early_stopping
    early_stopping_patience = args.early_stopping_patience
    batch_normalization = args.batch_normalization
    learning_rate = args.learning_rate
    dropout = args.dropout
    without_static = args.without_static
    remove_sapsii = args.remove_sapsii
    static_ffn_depth = args.static_ffn_depth
    static_hidden_dim = args.static_hidden_dim
    use_sapsii_scores = args.use_sapsii_scores
except:
    #### Main Settings ####
    DATA_NAME = 'mimic34000' # dataset name
    TASK_NAME = 'icd9' # tasks - Mortality, length of stay and ICD9
    model_type = 0
    working_path = u'..'
    #######################
print 'DATA_NAME:', DATA_NAME, 'TASK_NAME:', TASK_NAME, 'working_path:', working_path, 'model_type:', model_type

# Settings for task, model, path, etc
data_path = os.path.join(working_path, 'data', DATA_NAME)
result_path = os.path.join(working_path, 'output', DATA_NAME, data_file_name.split('.')[0], str(label_type))
result_log_path = os.path.join(result_path, 'log')
model_path = os.path.join(working_path, 'model', DATA_NAME, data_file_name.split('.')[0], str(label_type))
for required_path in [result_path, result_log_path, model_path]:
    if not os.path.exists(required_path):
        os.makedirs(required_path)

data_file_pathname = os.path.join(data_path, data_file_name)
folds_file_pathname = os.path.join(data_path, folds_file_name)
folds_stat_file_pathname = os.path.join(data_path, folds_stat_file_name)

# Load all data and folds
data_file = np.load(data_file_pathname)
folds_file = np.load(folds_file_pathname)
folds_stat_file = np.load(folds_stat_file_pathname)

if  DATA_NAME[:6] == 'mimic3' or DATA_NAME[:6] == 'mimic2':
    data = data_file['ep_data']
    tdata = data_file['ep_tdata']
    adm_features = data_file['adm_features_all']
    adm_labels = data_file['adm_labels_all']
    N = len(data)    
    X_all = data
    X_t = tdata
    X_t_flatten = data
    attrs_selected = []
    if TASK_NAME == 'icd9':
        task_id = label_type
        y = data_file['y_icd9'][:,task_id]
        task_name_str = 'ICD9_{0}'.format(task_id)
        folds = folds_file['folds_ep_icd9_multi'][0]
        folds_stat = folds_stat_file['folds_ep_icd9_multi'][0]
        tsfstds = []
        if use_sapsii_scores:
            for tr, va, ts in folds[0]:
                tsfstds.append(SAPSIITransformer(np.concatenate((tr, va))))
        else:
            for serial, non_serial in folds_stat[0]:
                tsfstds.append(FoldsStandardizer(serial, non_serial))
        reptimes = min(len(folds), 5)
        tsfstdlist = []
        for t in xrange(reptimes):
            tsfstdlist.append(tsfstds)
        loss_func = 'binary_crossentropy'
        final_activation = 'sigmoid'
        # Make sure the label is 0/1 binary variable
        y = (y > 0).astype(dtype=theano.config.floatX)
    elif TASK_NAME == 'mor':
        # y = data_file['y_mor']
        y = adm_labels[:, label_type]
        task_name_str = 'Mortality'
        folds = folds_file['folds_ep_mor'][label_type]
        folds_stat = folds_stat_file['folds_ep_mor'][label_type]
        tsfstds = []
        if use_sapsii_scores:
            for tr, va, ts in folds[0]:
                tsfstds.append(SAPSIITransformer(np.concatenate((tr, va))))
        else:
            for serial, non_serial in folds_stat[0]:
                tsfstds.append(FoldsStandardizer(serial, non_serial))
        reptimes = min(len(folds), 5)
        tsfstdlist = []
        for t in xrange(reptimes):
            tsfstdlist.append(tsfstds)
        loss_func = 'binary_crossentropy'
        final_activation = 'sigmoid'
        # Make sure the label is 0/1 binary variable
        y = (y > 0).astype(dtype=theano.config.floatX)
    elif TASK_NAME == 'los':
        y = data_file['y_los'] / 60.0 # convert minute to hour
        task_name_str = 'Length_Of_Stay'
        folds = folds_file['folds_ep_mor'][0]
        folds_stat = folds_stat_file['folds_ep_mor'][0]
        tsfstds = []
        if use_sapsii_scores:
            for tr, va, ts in folds[0]:
                tsfstds.append(SAPSIITransformer(np.concatenate((tr, va))))
        else:
            for serial, non_serial in folds_stat[0]:
                tsfstds.append(FoldsStandardizer(serial, non_serial))
        reptimes = min(len(folds), 5)
        tsfstdlist = []
        for t in xrange(reptimes):
            tsfstdlist.append(tsfstds)
        loss_func = 'mean_squared_error'
        final_activation = 'linear'

#### End here for the common configs ####

X_s = adm_features

print('test modality %d' % label_type)

if model_type == 1:
    clfname = 'MMDL1_output_dim={0}_ffn_depth={1}_merge_depth={2}_batch_size={3}_nb_epoch={4}_EarlyStopping={5}_EarlyStopping_patience={6}_batch_normalization={7}_learning_rate={8}_dropout={9}'.format(fit_parameters[0], fit_parameters[1], fit_parameters[2], batch_size, nb_epoch, early_stopping, early_stopping_patience, batch_normalization, learning_rate, dropout)
    if use_sapsii_scores:
        clfname += '_USESAPSII'
    if without_static:
        clfname += '_WithoutStatic'
    print(clfname)
    print(result_log_path)
    test_basic_learner(
       [X_s, X_t], y, None,
       clf=HierarchicalMultimodal(
           # static = True,
           static = (not without_static),
           remove_sapsii=remove_sapsii,
           size_Xs= X_s.shape[1],
           temporal = True,
           number_modality = 1,
           size_of_modality = [X_t.shape[2]],
           td_of_modality = [X_t.shape[1]],
           type_MMDL = 1,
           fit_parameters=fit_parameters,
           EarlyStopping=early_stopping,
           batch_size=batch_size,
           nb_epoch=nb_epoch,
           logdir=os.path.join(result_log_path, clfname),
           final_activation=final_activation,
           loss=loss_func,
           EarlyStopping_patience=early_stopping_patience,
           batch_normalization=batch_normalization,
           learning_rate=learning_rate,
           dropout=dropout
       ),
        taskName=task_name_str,
        clfName=clfname,
        splits = folds,
        tsf=tsfstdlist,
    )  # MMD with FFN and LSTM (MMD1)

elif model_type == 2:
    X_static = np.genfromtxt(os.path.join(static_features_path), delimiter=',')
    sftsflist = []
    for trainidx, valididx, testidx in folds[0]:
        X_static_train = X_static[np.concatenate([trainidx, valididx]).astype(np.int).flatten(), :]
        tmean = np.nanmean(X_static_train, axis=0)
        tstd = np.nanstd(X_static_train, axis=0)
        sftsflist.append(StaticFeaturesStandardizer(train_mean=tmean, train_std=tstd))
    sftsflist2 = []
    for t in range(reptimes):
        sftsflist2.append(sftsflist)
    clfname = 'MMDL2_output_dim={0}_ffn_depth={1}_merge_depth={2}_batch_size={3}_nb_epoch={4}_EarlyStopping={5}_EarlyStopping_patience={6}_batch_normalization={7}_learning_rate={8}_dropout={9}'.format(
        fit_parameters[0], fit_parameters[1], fit_parameters[2], batch_size, nb_epoch, early_stopping,
        early_stopping_patience, batch_normalization, learning_rate, dropout)
    print(clfname)
    print(result_log_path)
    test_basic_learner(
        X_static, y, None,
        clf=FeedForwardNetwork(
            hidden_dim=static_hidden_dim,
            final_activation=final_activation,
            loss=loss_func,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            learning_rate=learning_rate,
            ffn_depth=static_ffn_depth,
            batch_normalization=batch_normalization
        ),
        taskName=task_name_str,
        clfName=clfname,
        splits=folds,
        tsf=sftsflist2,
    )