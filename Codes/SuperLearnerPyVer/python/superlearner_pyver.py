from SuPyLearner.supylearner.core import *
from sklearn import datasets, svm, linear_model, neighbors, svm, ensemble, neural_network
from sklearn import tree
import numpy as np
from argparse import ArgumentParser
import os
from pygam import LogisticGAM, LinearGAM
from pyearth import Earth
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import re


def load_data(datapath, foldpath, taskname, labelnum, subset, sltype):
    if sltype == 'sl1':
        inputname = 'input_sapsiisubscores'
    elif sltype == 'sl2':
        inputname = 'input'
    # outputname = 'output'
    if subset == 'all':
        nametail = '.csv'
    elif subset == 'cv':
        nametail = '_cv.csv'
    elif subset == 'mv':
        nametail = '_mv.csv'
    inputname += nametail
    # outputname += nametail
    inputarray = np.genfromtxt(os.path.join(datapath, inputname), delimiter=',')
    # outputarray = np.genfromtxt(os.path.join(datapath, outputname), delimiter=',')[:, labelnum]

    hrs = datapath.rstrip('/').split('/')[-2].split('_')[0].rstrip('hrs')
    outputfile = np.load(os.path.join(foldpath, 'imputed-normed-ep_1_%s.npz' % hrs))
    if taskname == 'mor':
        outputarray = outputfile['adm_labels_all'][:, labelnum]
    elif taskname == 'icd9':
        outputarray = outputfile['y_icd9'][:, labelnum]
    elif taskname == 'los':
        outputarray = outputfile['y_los'] / 60.0

    if taskname == 'mor':
        foldname = 'folds_ep_mor'
    elif taskname == 'icd9':
        foldname = 'folds_ep_icd9'
    elif taskname == 'los':
        foldname = 'folds_ep_mor'
    folds = np.load(os.path.join(foldpath, '5-folds.npz'))[foldname][labelnum][0]
    return inputarray, outputarray, folds

def get_algolib_classification():
    return [
        linear_model.LogisticRegression(),
        # linear_model.LassoLarsIC(criterion='aic'),
        # LinearGAM(),
        linear_model.ElasticNet(),
        # Earth(),
        # linear_model.BayesianRidge(),
        ensemble.GradientBoostingClassifier(),
        neural_network.MLPClassifier(),
        ensemble.BaggingClassifier(),
        # tree.DecisionTreeClassifier(),
        ensemble.RandomForestClassifier(),
        # bart
        # ensemble.GradientBoostingClassifier(),
        # XGBClassifier(),
        # LGBMClassifier(),
    ], [
        'SL.glm',
        # 'SL.stepAIC',
        # 'SL.gam',
        'SL.glmnet',
        # 'SL.polymars',
        # 'SL.bayesglm',
        'SL.gbm',
        'SL.nnet',
        'SL.ipredbagg',
        # 'SL.rpartPrune',
        'SL.randomForest',
        # 'SL.bart',
        # 'GBDT',
        # 'XGBoost',
        # 'LightGBM'
    ]

def get_algolib_regression():
    return [
        linear_model.LinearRegression(),
        # linear_model.LassoLarsIC(criterion='aic'),
        # LinearGAM(),
        linear_model.ElasticNet(),
        # Earth(),
        # linear_model.BayesianRidge(),
        ensemble.GradientBoostingRegressor(),
        neural_network.MLPRegressor(),
        ensemble.BaggingRegressor(),
        # tree.DecisionTreeClassifier(),
        ensemble.RandomForestRegressor(),
        # bart
        # ensemble.GradientBoostingClassifier(),
        # XGBClassifier(),
        # LGBMClassifier(),
    ], [
        'SL.glm',
        # 'SL.stepAIC',
        # 'SL.gam',
        'SL.glmnet',
        # 'SL.polymars',
        # 'SL.bayesglm',
        'SL.gbm',
        'SL.nnet',
        'SL.ipredbagg',
        # 'SL.rpartPrune',
        'SL.randomForest',
        # 'SL.bart',
        # 'GBDT',
        # 'XGBoost',
        # 'LightGBM'
    ]

def main():
    parser = ArgumentParser()
    parser.add_argument('datapath',
                        help='path of data folder')
    parser.add_argument('foldpath',
                        help='path of fold file')
    parser.add_argument('taskname',
                        help='name of task, like mor/icd9/los')
    parser.add_argument('labelnum', type=int,
                        help='number of label used for current task')
    parser.add_argument('subset',
                        help='choose to use full dataset or only cv/mv, value must be all/cv/mv')
    parser.add_argument('sltype', default='sl2',
                        help='type of superlearner, sl1 uses sapsii scores, sl2 uses features')
    # parser.add_argument('modeltype', default='classification',
    #                     help='run classification task or regression task')
    args = parser.parse_args()
    datapath = args.datapath
    foldpath = args.foldpath
    taskname = args.taskname
    labelnum = args.labelnum
    subset = args.subset
    sltype = args.sltype
    assert subset in ['all', 'cv', 'mv']
    assert sltype in ['sl1', 'sl2']

    if taskname == 'los':
        modeltype = 'regression'
    else:
        modeltype = 'classification'

    X, y, folds = load_data(datapath, foldpath, taskname, labelnum, subset, sltype)

    if modeltype == 'classification':
        algolib, algonames = get_algolib_classification()
        sl = SuperLearner(algolib, algonames, loss='nloglik', K=10)
        metricname, metric = 'aurocs', roc_auc_score
    elif modeltype == 'regression':
        algolib, algonames = get_algolib_regression()
        sl = SuperLearner(algolib, algonames, loss='L2', coef_method='NNLS', K=10)
        # def scaled_mean_squared_error(y_true, y_pred):
        #     return mean_squared_error(y_true, y_pred)
        metricname, metric = 'mses', mean_squared_error

    risk_cv, y_pred_cv, y_true_cv = cv_superlearner(sl, X, y, K=5, fixed_folds=folds)

    metric_results = []
    for y_pred, y_true in zip(y_pred_cv, y_true_cv):
        metric_results.append(metric(y_true, y_pred[:,-1].flatten()))
    print(metric_results)
    print(np.mean(metric_results))
    print(np.std(metric_results))

    if modeltype == 'classification':
        np.savez(os.path.join(datapath, 'pyslresults-{0}-{1}-{2}-{3}.npz'.format(taskname, labelnum, subset, sltype)),
                risk_cv=risk_cv, y_pred_cv=y_pred_cv, y_true_cv=y_true_cv, aurocs=metric_results)
    elif modeltype == 'regression':
        np.savez(os.path.join(datapath, 'pyslresults-{0}-{1}-{2}-{3}.npz'.format(taskname, labelnum, subset, sltype)),
                 risk_cv=risk_cv, y_pred_cv=y_pred_cv, y_true_cv=y_true_cv, mses=metric_results)


if __name__ == '__main__':
    main()

    #Generate a dataset.
    # np.random.seed(100)
    # X, y=datasets.make_friedman1(1000)
    #
    # ols=linear_model.LinearRegression()
    # elnet=linear_model.ElasticNetCV()
    # ridge=linear_model.RidgeCV()
    # lars=linear_model.LarsCV()
    # lasso=linear_model.LassoCV()
    # nn=neighbors.KNeighborsRegressor()
    # svm1=svm.SVR()
    # svm2=svm.SVR(kernel='poly')
    # lib=[ols, elnet, ridge,lars, lasso, nn, svm1, svm2]
    # libnames=["OLS", "ElasticNet", "Ridge", "LARS", "LASSO", "kNN", "SVM rbf", "SVM poly"]
    #
    # sl=SuperLearner(lib, libnames, loss="L2")
    #
    # sl.fit(X, y)
    #
    # sl.summarize()
