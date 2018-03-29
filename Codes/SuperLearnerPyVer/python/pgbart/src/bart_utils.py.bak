#!/usr/bin/env python
# contains a bunch of functions that are shared across multiple files


import sys
import math
import optparse
import cPickle as pickle
import numpy as np
import random
import time
from itertools import izip
from scipy.special import gammaln, digamma
from scipy.special import gdtrc         # required only for regression
from scipy.optimize import fsolve       # required only for regression
import scipy.stats
from copy import copy


def logsumexp(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= tmp_max
    op = np.log(np.sum(np.exp(tmp))) + tmp_max
    return op


def softmax(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= float(tmp_max)
    tmp = np.exp(tmp)
    op = tmp / np.sum(tmp)
    return op


def assert_no_nan(mat, name='matrix'):
    try:
        assert(not any(np.isnan(mat)))
    except AssertionError:
        print '%s contains NaN' % name
        print mat
        raise AssertionError

def check_if_one(val):
    try:
        assert(np.abs(val - 1) < 1e-12)
    except AssertionError:
        print 'val = %s (needs to be equal to 1)' % val
        raise AssertionError

def check_if_zero(val):
    try:
        assert(np.abs(val) < 1e-10)
    except AssertionError:
        print 'val = %s (needs to be equal to 0)' % val
        raise AssertionError


def linear_regression(x, y):
    ls = np.linalg.lstsq(x, y)
    #print ls
    coef = ls[0]
    if ls[1]:
        sum_squared_residuals = float(ls[1])    # sum of squared residuals
    else:
        sum_squared_residuals = np.sum(np.dot(x, coef) - y)    # sum of squared residuals
    return (coef, sum_squared_residuals)


def sample_multinomial(prob):
    try:
        k = int(np.where(np.random.multinomial(1, prob, size=1)[0]==1)[0])
    except TypeError:
        print 'problem in sample_multinomial: prob = '
        print prob
        raise TypeError
    except:
        raise Exception
    return k


def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = int(np.sum(s > scores_cumsum))
    return k


class empty(object):
    def __init__(self):
        pass


def parser_add_common_options():
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset', default='toy-hypercube-3',
            help='name of the dataset [default: %default]')
    parser.add_option('--data_path', dest='data_path', default='../../process_data/',
            help='path of the dataset [default: %default]')
    parser.add_option('--debug', dest='debug', default='0', type='int',
            help='debug or not? (0=False, 1=True) [default: %default]')
    parser.add_option('--center_y', dest='center_y', default='0', type='int',
            help='do you want to center y at mean of training labels? (0=False, 1=True) [default: %default]')
    parser.add_option('--op_dir', dest='op_dir', default='results', 
            help='output directory for pickle files (NOTE: make sure directory exists) [default: %default]')
    parser.add_option('--tag', dest='tag', default='', 
            help='additional tag to identify results from a particular run [default: %default]')
    parser.add_option('--save', dest='save', default=0, type='int',
            help='do you wish to save the results? (1=True, 0=False) [default: %default]') 
    parser.add_option('-v', '--verbose',dest='verbose', default=1, type='int',
            help='verbosity level (0 is minimum, 4 is maximum) [default: %default]')
    parser.add_option('--init_id', dest='init_id', default=1, type='int',
            help='init_id (changes random seed for multiple initializations) [default: %default]')
    parser.add_option('--sample_y', dest='sample_y', default=0, type='int',
            help='do you want to sample the labels (successive conditional simulator in "Getting it right")? (1/0) [default: %default]')
    parser.add_option('--store_every_iteration', dest='store_every_iteration', default=0, type='int',
            help='do you want to store predictions at every iteration and their performance measures? (1/0) [default: %default]')
    parser.add_option('--n_iterations', dest='n_iterations', default=2000, type='int',
            help='number of MCMC iterations [default: %default]')
    parser.add_option('--n_run_avg', dest='n_run_avg', default=10, type='int',
            help='number of iterations after which the cumulative prediction is dumped out [default: %default]')
    parser.add_option('--store_all_stats', dest='store_all_stats', default=0, type='int',
            help='do you want to store all stats? (might take too much disk space) (1/0) [default: %default]')
    return parser


def parser_check_common_options(parser, settings):
    fail(parser, not(settings.save==0 or settings.save==1), 'save needs to be 0/1')
    fail(parser, not(settings.debug==0 or settings.debug==1), 'debug needs to be 0/1')
    fail(parser, not(settings.center_y==0 or settings.center_y==1), 'center_y needs to be 0/1')
    fail(parser, settings.n_iterations < 1, 'number of iterations needs to be >= 1')
    fail(parser, not(settings.sample_y==0), 'sample_y needs to be 0')
    fail(parser, not(settings.store_every_iteration==0 or settings.store_every_iteration==1), 'store_every_iteration needs to be 0/1')
    fail(parser, not(settings.store_all_stats==0 or settings.store_all_stats==1), 'store_all_stats needs to be 0/1')


def parser_add_tree_prior_hyperparameter_options(parser):
    group = optparse.OptionGroup(parser, "Prior specification: Hyperparameters of BART model")
    group.add_option('--alpha_split', dest='alpha_split', default=0.95, type='float',
            help='alpha-split for cgm tree prior  [default: %default]')   
    group.add_option('--beta_split', dest='beta_split', default=0.5, type='float',
            help='beta_split for cgm tree prior [default: %default]')    
    group.add_option('--alpha_bart', dest='alpha_bart', default=3.0, type='float',
            help='alpha_bart is the df parameter in BART [default: %default]')  # they try just 3 and 10
    group.add_option('--k_bart', dest='k_bart', default=2.0, type='float',
            help='k_bart controls the prior over mu (mu_prec) in BART [default: %default]')
    group.add_option('--q_bart', dest='q_bart', default=0.9, type='float',
            help='q_bart controls the prior over sigma^2 in BART [default: %default]')
    group.add_option('--m_bart', dest='m_bart', default=1, type='int',
            help='m_bart specifies the number of trees in BART [default: %default]')
    group.add_option('--variance', dest='variance', default='unconditional',
            help='which variance should you use for setting hyperparameters? (unconditional/leastsquares) [default: %default]')
    parser.add_option_group(group)
    return parser


def parser_check_tree_prior_hyperparameter_options(parser, settings):
    fail(parser, not(settings.alpha_split <= 1 and settings.alpha_split >= 0), 'alpha_split needs to be in [0, 1]')
    fail(parser, not(settings.beta_split > 0), 'beta_split needs to be > 0')
    fail(parser, not(settings.alpha_bart > 0), 'alpha_bart needs to be > 0')
    fail(parser, not(settings.k_bart > 0), 'k_bart needs to be > 0')
    fail(parser, not(settings.q_bart > 0), 'q_bart needs to be > 0')
    fail(parser, not(settings.m_bart > 0), 'm_bart needs to be > 0')
    fail(parser, not((settings.variance == 'unconditional') or (settings.variance == 'leastsquares')), \
            'variance needs to be unconditional or leastsquares')


def parser_add_smc_options(parser):
    group = optparse.OptionGroup(parser, "SMC options")
    group.add_option('--n_particles', dest='n_particles', default=10, type='int',
            help='number of particles [default: %default]')
    parser.add_option_group(group)
    # I have tested the code only with the default values below, 
    # but I left them in incase you want to modify the code later (remember to update filename for saving)
    group = optparse.OptionGroup(parser, "Not-so-optional SMC options",
            "I have tested the code only with the default values for these options, " 
            "but I left them incase you want to modify the code")
    group.add_option('--resample', dest='resample', default='multinomial',
            help='resampling method (multinomial/systematic) [default: %default]')
    group.add_option('--proposal', dest='proposal', default='prior',
            help='proposal (prior) [default: %default]')
    group.add_option('--ess_threshold', dest='ess_threshold', default=1.0, type='float',
            help='ess_threshold [default: %default]')
    group.add_option('--min_size', dest='min_size', default=1, type='int',
            help='minimum number of data points at leaf nodes [default: %default]')
    parser.add_option_group(group)
    return parser


def parser_check_smc_options(parser, settings):
    fail(parser, settings.n_particles < 1, 'number of particles needs to be >= 1')
    fail(parser, settings.min_size < 1, 'min_size needs to be >= 1')
    fail(parser, settings.ess_threshold < 0 or settings.ess_threshold > 1, 'ess_threshold needs to be in [0, 1]')
    fail(parser, (settings.resample != 'multinomial') and (settings.resample != 'systematic'), 
            'unknown resample (valid = multinomial/systematic)')
    fail(parser, (settings.proposal != 'prior'), 'unknown proposal (valid = prior)')


def parser_add_mcmc_options(parser):
    group = optparse.OptionGroup(parser, "MCMC options")
    group.add_option('--mcmc_type', dest='mcmc_type', default='pg',
                      help='type of MCMC (cgm/growprune/pg) [default: %default]')
    group.add_option('--init_pg', dest='init_pg', default='empty',
                      help='type of init for Particle Gibbs (empty/smc) [default: %default]')
    group.add_option('--init_mcmc', dest='init_mcmc', default='empty',
                      help='type of init for MCMC (empty/random) [default: %default]')
    parser.add_option_group(group)
    return parser


def parser_check_mcmc_options(parser, settings):
    fail(parser, not(settings.mcmc_type=='cgm' or settings.mcmc_type == 'growprune' or settings.mcmc_type=='pg'), \
            'mcmc_type needs to be cgm/growprune/pg')
    fail(parser, not(settings.init_pg=='empty' or settings.init_pg=='smc'), \
            'init_pg needs to be empty/smc')
    fail(parser, not(settings.init_mcmc=='empty' or settings.init_mcmc=='random'), \
            'init_mcmc needs to be empty/random')


def fail(parser, condition, msg):
    if condition:
        print msg
        print
        parser.print_help()
        sys.exit(1)


def add_stuff_2_settings(settings):
    settings.perf_dataset_keys = ['train', 'test']
    settings.perf_store_keys = ['pred_mean', 'pred_prob']
    settings.perf_metrics_keys = ['log_prob', 'mse']


def process_command_line():
    parser = parser_add_common_options()
    parser = parser_add_tree_prior_hyperparameter_options(parser)
    parser = parser_add_smc_options(parser)
    parser = parser_add_mcmc_options(parser)
    settings, args = parser.parse_args()
    add_stuff_2_settings(settings) 
    parser_check_common_options(parser, settings)
    parser_check_tree_prior_hyperparameter_options(parser, settings)
    parser_check_smc_options(parser, settings)
    parser_check_mcmc_options(parser, settings)
    return settings


def reset_random_seed(settings):
    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    random.seed(settings.init_id * 1000)


def check_dataset(settings):
    regression_datasets = set([])   # add regression dataset name here
    special_cases = settings.dataset[:3] == 'toy' or settings.dataset[:4] == 'test' \
            or settings.dataset[:8] == 'friedman' or settings.dataset[:4] == 'rsyn' \
            or settings.dataset[:8] == 'ctslices' or settings.dataset[:3] == 'msd' \
            or settings.dataset[:6] == 'houses'
    if not special_cases:
        try:
            assert(settings.dataset in regression_datasets)
        except AssertionError:
            print 'Invalid dataset for regression; dataset = %s' % settings.dataset
            raise AssertionError
    return special_cases


def load_data(settings):
    data = {}
    special_cases = check_dataset(settings)
    if not special_cases:
        data = pickle.load(open(settings.data_path + settings.dataset + '/' + settings.dataset + '.p', \
            "rb"))
    elif settings.dataset == 'toy-wtw':
        data = load_toy_wtw()
    elif settings.dataset[:13] == 'toy-hypercube':
        n_dim = int(settings.dataset[14:])
        data = load_toy_hypercube(n_dim, settings)
    elif settings.dataset[:8] == 'friedman':
        pos = [i for i, ch in enumerate(settings.dataset) if ch == '-']
        assert len(pos) == 1 or len(pos) == 4     # expect n_test to be present if n_train is present
        if len(pos) == 1:
            n_dim = int(settings.dataset[9:])
            n_train = 100
            n_test = 100
            variance = 0.1
        else:
            n_dim = int(settings.dataset[9:pos[1]])
            n_train = int(settings.dataset[pos[1]+1:pos[2]])
            n_test = int(settings.dataset[pos[2]+1:pos[3]])
            variance = float(settings.dataset[pos[3]+1:])
        print 'n_dim = %s, n_train = %s, n_test = %s, variance = %s' % (n_dim, n_train, n_test, variance)
        try:
            assert(n_dim >= 5)
        except AssertionError:
            print "For friedman dataset, dataset should be of the form friedman-dim (note '-') where dim >= 5"
            raise AssertionError
        data = load_friedman_data(n_dim, n_train, n_test, variance)
    elif settings.dataset[:4] == 'rsyn' or settings.dataset[:8] == 'ctslices' \
            or settings.dataset[:6] == 'houses' or settings.dataset[:3] == 'msd':
        data = load_rgf_datasets(settings)
    else:
        print 'Unknown dataset: ' + settings.dataset
        raise Exception
    assert(not data['is_sparse'])
    return data


def load_rgf_datasets(settings):
    filename_train = settings.data_path + 'exp-data' + '/' + settings.dataset
    filename_test = filename_train[:-3]
    x_train = np.loadtxt(filename_train + '.train.x')
    y_train = np.loadtxt(filename_train + '.train.y')
    x_test = np.loadtxt(filename_test + '.test.x')
    y_test = np.loadtxt(filename_test + '.test.y')
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    n_dim = x_train.shape[1]
    data = {'x_train': x_train, 'y_train': y_train, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def gen_wtw_data():
    # synthetic dataset described in "Bayesian CART: Prior Specification and Posterior Simulation"
    n_points = 300
    n_dim = 3
    variance = 0.25
    sd = math.sqrt(variance)
    x = np.zeros((n_points, n_dim))
    idx_first200 = np.arange(200)
    idx_last100 = np.arange(200, 300)
    idx_first100 = np.arange(100)
    idx_middle100 = np.arange(100, 200)
    x[idx_first200, 0] = np.random.uniform(0.1, 0.4, 200)
    x[idx_last100, 0] = np.random.uniform(0.6, 0.9, 100)
    x[idx_first100, 1] = np.random.uniform(0.1, 0.4, 100)
    x[idx_middle100, 1] = np.random.uniform(0.6, 0.9, 100)
    x[idx_last100, 1] = np.random.uniform(0.1, 0.9, 100)
    x[idx_first200, 2] = np.random.uniform(0.6, 0.9, 200)
    x[idx_last100, 2] = np.random.uniform(0.1, 0.4, 100)
    y = np.zeros(n_points)
    f = np.zeros(n_points)
    idx1 = np.logical_and(x[:, 0] <= 0.5, x[:, 1] <= 0.5)
    idx3 = np.logical_and(x[:, 0] <= 0.5, x[:, 1] > 0.5)
    idx5 = x[:, 0] > 0.5
    f[idx1] = 1
    f[idx3] = 3
    f[idx5] = 5
    y = f + sd * np.random.randn(n_points)
    return (x, y, f)


def load_toy_wtw():
    x, y, f = gen_wtw_data()
    n_train = n_test = 300
    n_dim = 3
    data = {'x_train': x, 'y_train': y, \
            'f_train': f, 'f_test': f, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x, \
            'y_test': y, 'n_test': n_test, 'is_sparse': False}
    return data


def gen_hypercube_data(n_points, n_dim, f_values=None):
    """
    synthetic hypercube-like dataset 
    x-values of data points are close to vertices of a hypercube
    y-value of data point is different
    f-values denotes the mean at each of the vertices 
    f-values generated only for training data; re-used for test data
    """
    y_sd = 0.
    x_sd = 0.1 
    mag = 3
    x = x_sd * np.random.randn(n_points, n_dim)
    n_vertices = 2 ** n_dim
    #i = np.random.randint(0, n_vertices, n_points)
    i = np.arange(n_vertices).repeat(n_points / n_vertices)     # equal distribution
    offsets = np.zeros((n_vertices, n_dim))
    for d in range(n_dim):
        tmp = np.ones(2**(n_dim-d))
        tmp[:2**(n_dim-d-1)] = -1
        offsets[:, d] = np.tile(tmp, (1, 2**d))[0]
    x += offsets[i, :]
    y = np.zeros(n_points)
    if f_values is None:
        # generate only for training data
        f_values = np.random.randn(n_vertices) * mag
    f = f_values[i]
    y = f + y_sd * np.random.randn(n_points)
    return (x, y, f, f_values)


def load_toy_hypercube(n_dim, settings):
    n_train = n_test = 10 * (2 ** n_dim)
    reset_random_seed(settings)
    x_train, y_train, f_train, f_values = gen_hypercube_data(n_train, n_dim)
    x_test, y_test, f_test, f_values = gen_hypercube_data(n_test, n_dim, f_values)
    data = {'x_train': x_train, 'y_train': y_train, \
            'f_train': f_train, 'f_test': f_test, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def gen_friedman_data(n_dim, n_points, variance):
    x = np.random.rand(n_points, n_dim)
    e = math.sqrt(variance) * np.random.randn(n_points)
    f = 10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * ((x[:, 2] - 0.5) ** 2) \
            + 10 * x[:,3] + 5 * x[:,4]
    y = f + e
    return (x, y, f)


def load_friedman_data(n_dim=5, n_train=100, n_test=100, variance=1):
    x_train, y_train, f_train = gen_friedman_data(n_dim, n_train, variance)
    x_test, y_test, f_test = gen_friedman_data(n_dim, n_test, variance)
    print 'mse with ground truth labels i.e. mean((y-f)^2): train = %.3f' % (np.mean((y_train - f_train) ** 2))
    print 'mse with ground truth labels i.e. mean((y-f)^2): test = %.3f' % (np.mean((y_test - f_test) ** 2))
    data = {'x_train': x_train, 'y_train': y_train, \
            'f_train': f_train, 'f_test': f_test, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def get_parent_id(node_id):
    if node_id == 0:
        op = 0
    else:
        op = int(math.ceil(node_id / 2.) - 1)
    return op


def get_sibling_id(node_id):
    if node_id == 0:
        op = 0
    else:
        parent = get_parent_id(node_id)
        left, right = get_children_id(parent)
        if left == node_id:
            op = right
        else:
            op = left
    return op


def get_depth(node_id):
    op = int(math.floor(math.log(node_id + 1, 2)))
    return op


def get_children_id(node_id):
    tmp = 2 * (node_id + 1)
    return (tmp - 1, tmp)


class Param(object):
    def __init__(self, settings):
        self.alpha_split = settings.alpha_split
        self.beta_split = settings.beta_split


def get_filename_bart(settings):
    param_str = 'bart-%s_%s_%s_%s' % (settings.alpha_bart, settings.k_bart, settings.q_bart, settings.m_bart)
    split_str = 'cgm-%s_%s' % (settings.alpha_split, settings.beta_split)
    if settings.mcmc_type == 'pg':
        pg_settings = '-C-%s-ess-%s-%s-%s-nodewise-%s' % \
            (settings.n_particles, settings.ess_threshold, \
             settings.proposal, settings.resample, settings.init_pg)
    else:
        pg_settings = ''
    filename = settings.op_dir + '/' + '%s-tree_prior-%s-param-%s-n_iter-%s' \
            '-init_id-%s-mcmc-%s%s-sample_y-%d-center_y-%d-variance-%s-tag-%s.p' % \
            (settings.dataset, split_str, param_str, settings.n_iterations,\
             settings.init_id, settings.mcmc_type, pg_settings, settings.sample_y, \
             settings.center_y, settings.variance, settings.tag)
    return filename


class Tree(object):
    def __init__(self, train_ids=np.arange(0, dtype='int'), param=empty(), settings=empty(), cache_tmp={}):
        self.depth = -1
        root_node_id = 0
        if cache_tmp:
            self.leaf_nodes = [root_node_id]
            self.non_leaf_nodes = []
            self.do_not_split = {root_node_id: False}
            self.sum_y = {root_node_id: cache_tmp['sum_y']}
            self.sum_y2 = {root_node_id: cache_tmp['sum_y2']}
            self.n_points = {root_node_id: cache_tmp['n_points']}
            self.loglik = {root_node_id: cache_tmp['loglik']}
            self.param_n = {root_node_id: cache_tmp['param_n']}
            self.train_ids = {root_node_id: train_ids} # NOTE: not copying train_ids at root since they are shared
            self.node_info = {}
            self.logprior = {root_node_id: math.log(self.compute_pnosplit(root_node_id, param))}
            self.loglik_current = self.loglik[root_node_id] + 0.0

    def prior_proposal(self, data, param, settings, cache, node_id, train_ids, log_psplit):
        """
        NOTE: logprior_nodeid computations have been switched off
        assume a uniform prior over the features; see how feat_id_chosen is sampled below
        """ 
        pnosplit = self.compute_pnosplit(node_id, param)    # pnosplit will be verified later
        do_not_split_node_id = np.random.rand(1) <= pnosplit
        split_not_supported = False
        if not do_not_split_node_id:
            np.random.shuffle(cache['range_n_dim_shuffle'])
            for feat_id_chosen in cache['range_n_dim_shuffle']:
                x_min, x_max, idx_min, idx_max, feat_score_cumsum_prior_current = \
                            self.get_info_dimension(data, cache, train_ids, settings, feat_id_chosen)
                if idx_min == idx_max:
                    split_not_supported = True
                    continue
                z_prior = feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min]
                prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                            feat_score_cumsum_prior_current[idx_min]) / z_prior
                idx_split_chosen = sample_multinomial(prob_split_prior)
                idx_split_global = idx_split_chosen + idx_min + 1
                split_chosen = cache['feat_idx2midpoint'][feat_id_chosen][idx_split_global]
                if settings.debug == 1:
                    is_split_valid(split_chosen, x_min, x_max)
                logprior_nodeid = 0.
                # logprior_nodeid is not really needed to sample
                # NOTE: using this incorrect value might result in invalid logprior, logprob values in BART
                (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                        compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, split_chosen, settings)
                if settings.debug == 1:
                    try:
                        check_if_zero(cache_tmp['sum_y_left'] + cache_tmp['sum_y_right'] - self.sum_y[node_id])
                        check_if_zero(cache_tmp['sum_y2_left'] + cache_tmp['sum_y2_right'] - self.sum_y2[node_id])
                        check_if_zero(cache_tmp['n_points_left'] + cache_tmp['n_points_right'] - self.n_points[node_id])
                    except AssertionError:
                        print 'sum_y = %.2f, sum_y_left = %.2f, sum_y_right = %.2f, left+right = %.2f' % \
                            (self.sum_y[node_id], cache_tmp['sum_y_left'], cache_tmp['sum_y_right'], cache_tmp['sum_y_left']+cache_tmp['sum_y_right'])
                        print 'sum_y2 = %.2f, sum_y2_left = %.2f, sum_y2_right = %.2f, left+right = %.2f' % \
                            (self.sum_y2[node_id], cache_tmp['sum_y2_left'], cache_tmp['sum_y2_right'], cache_tmp['sum_y2_left']+cache_tmp['sum_y2_right'])
                        raise AssertionError
                if settings.verbose >= 2:
                    print 'loglik (of all data points in parent) = %.2f' % self.loglik[node_id]
                log_sis_ratio_loglik = loglik_left + loglik_right - self.loglik[node_id]
                log_sis_ratio_prior = 0.
                # contributions of feat_id and psplit cancel out for precomputed proposals
                log_sis_ratio = log_sis_ratio_loglik + log_sis_ratio_prior
                self.log_sis_ratio_d[node_id] = (log_sis_ratio_loglik, log_sis_ratio_prior)
                if settings.verbose >= 2:
                    print 'idx_split_chosen = %d, split_chosen = %f' % (idx_split_chosen, split_chosen)
                    print 'feat_id_chosen = %f' % (feat_id_chosen)
                split_not_supported = False
                break       # if you got this far, you deserve a break!
        if split_not_supported:
            do_not_split_node_id = True
            # pnosplit = 1.0
        if do_not_split_node_id:
            feat_id_chosen = -1
            split_chosen = 3.14
            idx_split_global = -1
            logprior_nodeid = 0.
            # commented out stuff below since logprior_nodeid is not really needed for PG
            #if split_not_supported:
            #    logprior_nodeid = 0.0
            #else:
            #    logprior_nodeid = np.log(pnosplit)
            log_sis_ratio = 0.0     # probability of not splitting under prior and proposal are both the same
            (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                init_left_right_statistics()
        return (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
            train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right)

    def update_left_right_statistics(self, cache_tmp, node_id, logprior_nodeid, train_ids_left,\
            train_ids_right, loglik_left, loglik_right, feat_id_chosen, split_chosen, idx_split_global, \
            settings, param, data, cache):
        left, right = get_children_id(node_id)
        self.logprior[node_id] = logprior_nodeid
        self.node_info[node_id] = [feat_id_chosen, split_chosen, idx_split_global]
        self.loglik[left] = loglik_left
        self.loglik[right] = loglik_right
        self.do_not_split[left] = stop_split(train_ids_left, settings, data, cache)
        self.do_not_split[right] = stop_split(train_ids_right, settings, data, cache)
        if self.do_not_split[left]:
            self.logprior[left] = 0.0
        else:
            self.logprior[left] = math.log(self.compute_pnosplit(left, param))
        if self.do_not_split[right]:
            self.logprior[right] = 0.0
        else:
            self.logprior[right] = math.log(self.compute_pnosplit(right, param))
        if settings.debug == 1:
            assert(cache_tmp['n_points_left'] > 0)
            assert(cache_tmp['n_points_right'] > 0)
        self.train_ids[left] = train_ids_left
        self.leaf_nodes.append(left)
        self.train_ids[right] = train_ids_right
        self.leaf_nodes.append(right)
        self.sum_y[left] = cache_tmp['sum_y_left']
        self.sum_y2[left] = cache_tmp['sum_y2_left']
        self.n_points[left] = cache_tmp['n_points_left']
        self.param_n[left] = cache_tmp['param_left']
        self.sum_y[right] = cache_tmp['sum_y_right']
        self.sum_y2[right] = cache_tmp['sum_y2_right']
        self.n_points[right] = cache_tmp['n_points_right']
        self.param_n[right] = cache_tmp['param_right']
        try:
            self.leaf_nodes.remove(node_id)
        except ValueError:
            pass
        self.non_leaf_nodes.append(node_id)
        self.depth = max(get_depth(left), self.depth)

    def remove_leaf_node_statistics(self, node_id, settings):
        try:
            self.leaf_nodes.remove(node_id)
        except:
            print '%s is not a leaf node' % node_id
            raise Exception
        self.loglik.pop(node_id)
        self.train_ids.pop(node_id)
        self.logprior.pop(node_id)
        self.sum_y.pop(node_id)
        self.sum_y2.pop(node_id)
        self.n_points.pop(node_id)
        self.param_n.pop(node_id)

    def sample_split_prior(self, data, param, settings, cache, node_id):
        train_ids = self.train_ids[node_id]
        n_train_ids = len(train_ids)
        log_psplit = math.log(self.compute_psplit(node_id, param))
        pnosplit = self.compute_pnosplit(node_id, param)    # pnosplit will be verified later
        feat_id_valid, score_feat, feat_split_info, split_not_supported \
                    = self.find_valid_dimensions(data, cache, train_ids, settings)
        if split_not_supported:
            do_not_split_node_id = True
            feat_id_chosen = -1
            split_chosen = 3.14     # i like pi :)
            idx_split_global = -1
            logprior_nodeid = 0.0
        else: 
            do_not_split_node_id = False
            feat_id_perm, n_feat_subset, log_prob_feat = \
                    score_features(settings, feat_id_valid, score_feat, split_not_supported)
            feat_id_chosen = sample_multinomial_scores(score_feat)   
            idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = \
                    feat_split_info[feat_id_chosen] 
            z_prior = feat_score_cumsum_prior_current[idx_max] - \
                    feat_score_cumsum_prior_current[idx_min]
            prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                        feat_score_cumsum_prior_current[idx_min]) / z_prior
            idx_split_chosen = sample_multinomial(prob_split_prior)
            idx_split_global = idx_split_chosen + idx_min + 1
            split_chosen = cache['feat_idx2midpoint'][feat_id_chosen][idx_split_global]
            if settings.debug == 1:
                is_split_valid(split_chosen, x_min, x_max)
            logprior_nodeid_tau = math.log(prob_split_prior[idx_split_chosen])
            logprior_nodeid = log_psplit + logprior_nodeid_tau \
                                + log_prob_feat[feat_id_chosen]
            if settings.verbose >= 2:
                print 'idx_split_chosen = %d, split_chosen = %f' % (idx_split_chosen, split_chosen)
                print 'feat_id_chosen = %f' % (feat_id_chosen)
            if settings.verbose >= 3:
                print '3 terms in sample_split_prior for node_id = %s; %s, %s, %s' \
                         % (node_id, log_psplit, logprior_nodeid_tau, log_prob_feat[feat_id_chosen])
                print 'feat_id = %s, idx_split_chosen = %d, split_chosen = %f' % (feat_id_chosen, idx_split_chosen, split_chosen)
                print 'log prob_split_prior = %s' % math.log(prob_split_prior)
                print
        return (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, logprior_nodeid)
    
    def find_valid_dimensions(self, data, cache, train_ids, settings):
        score_feat = cache['prob_feat']
        first_time = True
        if settings.verbose >= 3:
            print 'original score_feat = %s' % score_feat
        feat_split_info = {}
        for feat_id in cache['range_n_dim']:
            x_min, x_max, idx_min, idx_max, feat_score_cumsum_prior_current = \
                self.get_info_dimension(data, cache, train_ids, settings, feat_id)
            feat_score_cumsum_prior_current = cache['feat_score_cumsum_prior'][feat_id] 
            if settings.verbose >= 3:
                print 'x_min = %s, x_max = %s, idx_min = %s, idx_max = %s' % \
                        (x_min, x_max, idx_min, idx_max)
            if idx_min == idx_max:
                if first_time:          # lazy copy
                    score_feat = cache['prob_feat'].copy()
                    first_time = False
                score_feat[feat_id] = 0
            else:
                feat_split_info[feat_id] = [idx_min, idx_max, x_min, x_max, \
                        feat_score_cumsum_prior_current]
        feat_id_valid = [feat_id for feat_id in cache['range_n_dim'] if score_feat[feat_id] > 0]
        split_not_supported = (len(feat_id_valid) == 0)
        if settings.verbose >= 3:
            print 'in find_valid_dimensions now'
            print 'training data in current node =\n %s' % data['x_train'][train_ids, :]
            print 'score_feat = %s, feat_id_valid = %s' % (score_feat, feat_id_valid)
        return (feat_id_valid, score_feat, feat_split_info, split_not_supported)

    def recompute_prob_split(self, data, param, settings, cache, node_id):
        train_ids = self.train_ids_new[node_id]
        if stop_split(train_ids, settings, data, cache):
            self.logprior_new[node_id] = -np.inf
        else:
            feat_id_chosen, split_chosen, idx_split_global = self.node_info_new[node_id]
            feat_id_valid, score_feat, feat_split_info, split_not_supported \
                        = self.find_valid_dimensions(data, cache, train_ids, settings)
            if feat_id_chosen not in feat_id_valid:
                self.logprior_new[node_id] = -np.inf
            else:
                log_prob_feat = np.log(score_feat) - np.log(np.sum(score_feat))
                idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = \
                        feat_split_info[feat_id_chosen] 
                if (split_chosen <= x_min) or (split_chosen >= x_max):
                    self.logprior_new[node_id] = -np.inf
                else:
                    z_prior = feat_score_cumsum_prior_current[idx_max] - \
                            feat_score_cumsum_prior_current[idx_min]
                    prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                                feat_score_cumsum_prior_current[idx_min]) / z_prior
                    idx_split_chosen = idx_split_global - idx_min - 1
                    logprior_nodeid_tau = math.log(prob_split_prior[idx_split_chosen])
                    log_psplit = math.log(self.compute_psplit(node_id, param))
                    self.logprior_new[node_id] = log_psplit + logprior_nodeid_tau \
                                                 + log_prob_feat[feat_id_chosen]
                    if settings.verbose >= 3:
                        print '3 terms in recompute for node_id = %s; %s, %s, %s' \
                               % (node_id, log_psplit, logprior_nodeid_tau, \
                                                     log_prob_feat[feat_id_chosen])
                        print 'feat_id = %s, idx_split_chosen = %d, split_chosen = %f' % (feat_id_chosen, idx_split_chosen, split_chosen)
                        print 'log prob_split_prior = %s' % np.log(prob_split_prior)
                        print

    def get_info_dimension(self, data, cache, train_ids, settings, feat_id):
        x_min = np.min(data['x_train'][train_ids, feat_id])
        x_max = np.max(data['x_train'][train_ids, feat_id])
        idx_min = cache['feat_val2idx'][feat_id][x_min]
        idx_max = cache['feat_val2idx'][feat_id][x_max]
        feat_score_cumsum_prior_current = cache['feat_score_cumsum_prior'][feat_id] 
        return (x_min, x_max, idx_min, idx_max, feat_score_cumsum_prior_current)

    def print_tree(self):
        try:
            print 'leaf nodes are %s, non-leaf nodes are %s' % (self.leaf_nodes, self.non_leaf_nodes)
            print 'logprior = %s, loglik = %s' % (self.logprior, self.loglik)
            print 'sum(logprior) = %s, sum(loglik) = %s' % (self.compute_logprior(), self.compute_loglik())
        except:
            print 'leaf nodes are %s' % self.leaf_nodes
        print 'node_id\tdepth\tfeat_id\t\tsplit_point'
        #for node_id in self.non_leaf_nodes:
        for node_id in self.node_info:
            try:
                feat_id, split, idx_split_global = self.node_info[node_id]
            except (IndexError, ValueError):          # more than 2 values to unpack
                feat_id, split = -1, np.float('nan')
            print '%3d\t%3d\t%6d\t\t%.2f' % (node_id, get_depth(node_id), \
                    feat_id, split)

    def gen_rules_tree(self):
        """ 
        gen_rules_tree rewrites a tree as a set of rules to make eval easier 
        NOTE: x denotes features, leaf_id denotes node_id of the leaves
        """
        self.rules = 'leaf_id = np.zeros(x.shape[0], dtype="int"); '
        for node_id in self.leaf_nodes:
            if node_id == 0:
                s = ''
                break
            nid = copy(node_id)
            s = 'ind = '
            while nid != 0:
                pid = get_parent_id(nid)
                dec = '<=' if nid == (2 * pid + 1) else '>'
                s += '(x[:, %d] %s %f) & ' % (self.node_info[pid][0], dec, self.node_info[pid][1])
                nid = pid
            self.rules += s[:-3]
            self.rules += '; leaf_id[ind] = %d; ' % node_id

    def traverse(self, x):
        node_id = 0
        while True:
            if node_id in self.leaf_nodes:
                break
            left, right = get_children_id(node_id)
            feat_id, split, idx_split_global = self.node_info[node_id]
            if x[feat_id] <= split:
                node_id = left
            else:
                node_id = right
        return node_id

    def predict_real_val(self, x, param, settings):
        """
        aggregate prediction vals outside and use lambda_bart to compute posterior
        """
        pred_val = np.zeros(x_test.shape[0])
        for n, x_ in enumerate(x_test):
            node_id = self.traverse(x_)
            pred_val[n] = self.pred_val_n[node_id]
        return pred_val

    def predict_real_val_fast(self, x, param, settings):
        """
        faster version of predict_real_val
        """
        exec(self.rules)    # create variable "leaf_id"
        pred_val = self.pred_val_n[leaf_id]
        return pred_val

    def compute_psplit(self, node_id, param):
        return param.alpha_split * math.pow(1 + get_depth(node_id), -param.beta_split)
    
    def compute_pnosplit(self, node_id, param):
        return 1.0 - self.compute_psplit(node_id, param)
    
    def compute_loglik(self):
        tmp = [self.loglik[node_id] for node_id in self.leaf_nodes]
        return sum(tmp)
    
    def compute_logprior(self):
        tmp = sum([self.logprior[node_id] for node_id in self.leaf_nodes]) \
                + sum([self.logprior[node_id] for node_id in self.non_leaf_nodes])
        return tmp

    def compute_logprob(self):
        return self.compute_loglik() + self.compute_logprior()

    def update_loglik_node_all(self, data, param, cache, settings):
        for node_id in self.loglik:
            self.update_loglik_node(node_id, data, param, cache, settings)
    
    def update_loglik_node(self, node_id, data, param, cache, settings):
        sum_y, sum_y2, n_points = get_reg_stats(data['y_train'][self.train_ids[node_id]])
        self.sum_y[node_id], self.sum_y2[node_id] = sum_y, sum_y2
        self.loglik[node_id], self.param_n[node_id] = compute_normal_normalizer(sum_y, \
                sum_y2, n_points, param, cache, settings)

    def update_depth(self):
        max_leaf = max(self.leaf_nodes)
        self.depth = get_depth(max_leaf)

    def check_depth(self):
        max_leaf = max(self.leaf_nodes)
        depth = get_depth(max_leaf)
        try:
            assert self.depth == depth
        except AssertionError:
            if max_leaf != 0:
                print 'Error in check_depth: self.depth = %s, max_leaf = %s, depth(max_leaf) = %s' \
                        % (self.depth, max_leaf, depth)
                raise AssertionError


def compute_normal_posterior_bart(mu, lambda_bart):
    pred_mean = mu
    log_const = 0.5 * (math.log(lambda_bart) - math.log(2 * math.pi))
    pred_param = (mu, lambda_bart, log_const)
    return (pred_mean, pred_param)


def compute_metrics_regression(y_test, pred_mean, pred_prob):
    mse, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        mse += math.pow(pred_mean[n] - y, 2)
        log_tmp_pred = np.log(pred_prob[n])     # try weighted log-sum-exp?
        try:
            assert(not np.isinf(abs(log_tmp_pred)))
        except AssertionError:
            # print 'WARNING: abs(log_tmp_pred) = inf in compute_metrics_regression'
            # raise AssertionError
            pass
        log_prob += log_tmp_pred
    mse /= (n + 1)
    log_prob /= (n + 1)
    metrics = {'mse': mse, 'log_prob': log_prob}
    return metrics


def test_compute_metrics_regression():
    n = 100
    pred_prob = np.random.rand(n)
    y = np.random.randn(n)
    pred = np.ones(n)
    metrics = compute_metrics_regression(y, pred, pred_prob)
    print 'chk if same: %s, %s' % (metrics['mse'], np.mean((y - 1) ** 2))
    print 'chk if same: %s, %s' % (metrics['log_prob'], np.mean(np.log(pred_prob)))
    assert np.abs(metrics['mse'] - np.mean((y - 1) ** 2)) < 1e-3
    assert np.abs(metrics['log_prob'] - np.mean(np.log(pred_prob))) < 1e-3


def compute_nn_loglik(x, param_nn):
    (mu, prec, log_const) = param_nn
    op = - 0.5 * prec * ((x - mu) ** 2) + log_const
    return op


def is_split_valid(split_chosen, x_min, x_max):
    try:
        assert(split_chosen > x_min)
        assert(split_chosen < x_max)
    except AssertionError:
        print 'split_chosen <= x_min or >= x_max'
        raise AssertionError


def stop_split(train_ids, settings, data, cache):
    if (len(train_ids) <= settings.min_size):
        op = True
    else:
        op = no_valid_split_exists(data, cache, train_ids, settings)
    return op


def compute_nn_normalizer(sum_y, sum_y2, n_points, param, cache):
    mu_prec_post = param.lambda_bart * n_points + param.mu_prec
    mu_mean_post = (param.mu_prec * param.mu_mean + param.lambda_bart * sum_y) / mu_prec_post
    # see (42) in [M07] for the derivation of the marginal likelihood below
    op = cache['nn_prior_term'] - n_points * cache['half_log_2pi'] \
            + 0.5 * (n_points * param.log_lambda_bart - math.log(mu_prec_post) \
                     + mu_prec_post * mu_mean_post * mu_mean_post - param.lambda_bart * sum_y2)
    return (op, (mu_mean_post, mu_prec_post))


def test_compute_nn_normalizer(mu_mean=0, mu_prec=0.1, lambda_bart=1, n_points=10, n_sampl=10000):
    if True:
        mu = np.random.randn(1) / np.sqrt(mu_prec) + mu_mean
        y = np.random.randn(n_points) / np.sqrt(lambda_bart) + mu
    else:
        # more trivial example, but don't try this with n_points > 10
        mu = mu_mean
        y = np.ones(n_points) * mu
    print 'mu = %s' % (mu)
    sum_y = float(np.sum(y))
    sum_y2 = float(np.sum(y ** 2))
    param = empty()
    param.mu_mean = mu_mean 
    param.mu_prec = mu_prec
    param.lambda_bart = lambda_bart
    param.log_lambda_bart = math.log(lambda_bart)
    cache = {}
    cache['nn_prior_term'] = 0.5 * math.log(param.mu_prec) - 0.5 * param.mu_prec * param.mu_mean * param.mu_mean
    cache['half_log_2pi'] = 0.5 * math.log(2 * math.pi)
    log_marginal, post_param = compute_nn_normalizer(sum_y, sum_y2, n_points, param, cache)
    print 'log_marginal (analytical) = %s, parameters of posterior = %s' % (log_marginal, post_param)
    log_marg_samples = -np.inf * np.ones(n_sampl)
    for s in range(n_sampl):
        mu = np.random.randn(1) / np.sqrt(mu_prec) + mu_mean
        log_marg_samples[s] = 0.5 * n_points * (math.log(lambda_bart) - math.log(2 * math.pi)) - 0.5 * lambda_bart * np.sum((y - mu) ** 2)
    log_marginal_s = logsumexp(log_marg_samples) - math.log(n_sampl)
    print 'log_marginal (sampled) = %s' % log_marginal_s
    assert(np.abs(log_marginal_s - log_marginal) < 0.1)


def compute_normal_normalizer(sum_y, sum_y2, n_points, param, cache, settings):
    op, param = compute_nn_normalizer(sum_y, sum_y2, n_points, param, cache)
    return (op, param)


def compute_log_pnosplit_children(node_id, param):
    left, right = get_children_id(node_id)
    tmp = math.log(1 - param.alpha_split * math.pow(1 + get_depth(left), -param.beta_split)) \
            +  math.log(1 - param.alpha_split * math.pow(1 + get_depth(right), -param.beta_split))
    return tmp


def init_left_right_statistics():
    return(None, None, {}, -np.inf, -np.inf)


def score_features(settings, feat_id_valid, score_feat, split_not_supported):
    feat_id_perm = feat_id_valid
    n_feat = len(feat_id_perm)
    if split_not_supported:
        log_prob_feat = np.ones(score_feat.shape) * np.nan
    else:
        log_prob_feat = np.log(score_feat) - np.log(np.sum(score_feat))
        if (settings.debug == 1) and feat_id_perm:
            try:
                assert(np.abs(logsumexp(log_prob_feat)) < 1e-12)
            except AssertionError:
                print 'feat_id_perm = %s' % feat_id_perm
                print 'score_feat = %s' % score_feat
                print 'logsumexp(log_prob_feat) = %s (needs to be 0)' % logsumexp(log_prob_feat)
                raise AssertionError
    return (feat_id_perm, n_feat, log_prob_feat)


def compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, \
        split_chosen, settings, do_not_compute_loglik=False):
    cond = data['x_train'][train_ids, feat_id_chosen] <= split_chosen
    train_ids_left = train_ids[cond]
    train_ids_right = train_ids[~cond]
    cache_tmp = {}
    sum_y_left = float(np.sum(data['y_train'][train_ids_left]))
    sum_y2_left = float(np.sum(data['y_train'][train_ids_left] ** 2))
    n_points_left = len(train_ids_left)
    loglik_left, param_left = compute_normal_normalizer(sum_y_left, sum_y2_left, n_points_left, param, cache, settings)
    cache_tmp['sum_y_left'] = sum_y_left
    cache_tmp['sum_y2_left'] = sum_y2_left
    cache_tmp['n_points_left'] = n_points_left
    cache_tmp['param_left'] = param_left
    sum_y_right = float(np.sum(data['y_train'][train_ids_right]))
    sum_y2_right = float(np.sum(data['y_train'][train_ids_right] ** 2))
    n_points_right = len(train_ids_right)
    loglik_right, param_right = compute_normal_normalizer(sum_y_right, sum_y2_right, n_points_right, param, cache, settings)
    cache_tmp['sum_y_right'] = sum_y_right
    cache_tmp['sum_y2_right'] = sum_y2_right
    cache_tmp['n_points_right'] = n_points_right
    cache_tmp['param_right'] = param_right
    if settings.verbose >= 2:
        print 'feat_id_chosen = %s, split_chosen = %s' % (feat_id_chosen, split_chosen)
        print 'y (left) = %s\ny (right) = %s' % (data['y_train'][train_ids_left], \
                                                    data['y_train'][train_ids_right])
        print 'loglik (left) = %.2f, loglik (right) = %.2f' % (loglik_left, loglik_right)
    return(train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right)


def precompute(data, settings):
    param = Param(settings)
    cache_tmp = {}
    # BART prior
    param.m_bart = settings.m_bart
    param.k_bart = settings.k_bart
    # See section 2.2.3 in page 271 of BART paper for how these parameters are set
    #
    # mu_mean, mu_prec are priors over gaussian in leaf node of each individual tree
    # Basic idea is to choose mu_mean and mu_prec such that the interval [y_min, y_max] (see below) 
    #       contains E[Y|X=x] with a specified probability (decided by k_bart: prob=0.95 if k_bart=2)
    # y_min = m_bart * mu_mean - k_bart * sqrt(m_bart / mu_prec) 
    # y_max = m_bart * mu_mean + k_bart * sqrt(m_bart / mu_prec) 
    y_max = np.max(data['y_train'])
    y_min = np.min(data['y_train'])
    y_diff = y_max - y_min
    param.mu_mean = float((y_min + 0.5 * y_diff) / param.m_bart)     # robust to outliers?
    # param.mu_mean = np.mean(data['y_train']) / param.m_bart
    param.mu_prec = float(param.m_bart * (2 * param.k_bart / y_diff) ** 2)
    mu_sd = 1.0 / math.sqrt(param.mu_prec)
    print 'y_diff = %.3f, y_min = %.3f, y_max = %.3f' % (y_diff, y_min, y_max)
    print 'mean(y) = %.3f' % np.mean(data['y_train'])
    print 'mu_mean = %.3f, mu_prec = %.3f, mu_sd = %.3f' % (param.mu_mean, param.mu_prec, mu_sd)
    print 'mu_mean - mu_sd = %.3f, mu_mean + mu_sd = %.3f' % (param.mu_mean - mu_sd, param.mu_mean + mu_sd)
    tmp_mean, tmp_stddev = param.mu_mean * param.m_bart, param.k_bart * math.sqrt(param.m_bart / param.mu_prec)
    print 'tmp_mean = %.3f, tmp_stddev = %.3f' % (tmp_mean, tmp_stddev)
    print 'expected CI (for k_bart = %.3f) = (%.3f, %.3f)' % (param.k_bart, tmp_mean-tmp_stddev, tmp_mean+tmp_stddev)
    #
    # See section 2.2.4 in page 272 of BART paper for how these parameters are set
    prec_unconditional = 1.0 / np.var(data['y_train'])
    print 'unconditional variance = %.3f, prec = %.3f' % (1.0 / prec_unconditional, prec_unconditional)
    if settings.variance == "leastsquares":
        ls_coef, ls_sum_squared_residuals = linear_regression(data['x_train'], data['y_train'])
        ls_var = ls_sum_squared_residuals / (data['n_train'] - 1)
        prec = 1.0 / ls_var
        print 'least squares variance = %.3f, prec = %.3f' % (ls_var, prec)
        if prec < prec_unconditional:
            print 'least squares variance seems higher than unconditional ... something is weird'
    else:
        print 'WARNING: lambda_bart initialized to unconditional precision'
        prec = prec_unconditional
    param.alpha_bart = settings.alpha_bart
    param.beta_bart = compute_gamma_param(prec, param.alpha_bart, settings.q_bart)
    # ensures that 1-gamcdf(prec; shape=alpha_bart, rate=beta_bart) \approx settings.q_bart 
    # i.e. all values of precision are higher than the unconditional variance of Y
    #param.lambda_bart = param.alpha_bart / param.beta_bart      #FIXME: better init? check sensitivity
    if settings.variance == "leastsquares":
        param.lambda_bart = float(prec)
    else:
        param.lambda_bart = float(prec) * 2   # unconditional precision might be too pessimistic
    print 'unconditional precision = %.2f, initial lambda_bart = %.2f' % (prec_unconditional, param.lambda_bart)
    settings.lambda_bart = param.lambda_bart
    param.log_lambda_bart = math.log(param.lambda_bart)
    cache_tmp['nn_prior_term'] = 0.5 * math.log(param.mu_prec) - 0.5 * param.mu_prec * param.mu_mean * param.mu_mean
    cache_tmp['n_points'] = data['n_train']
    cache_tmp['half_log_2pi'] = 0.5 * math.log(2 * math.pi)
    cache_tmp['sum_y'] = float(np.sum(data['y_train']) / settings.m_bart)
    cache_tmp['sum_y2'] = float(np.sum((data['y_train'] /settings.m_bart) ** 2))
    op_tmp, param_tmp = compute_normal_normalizer(cache_tmp['sum_y'], cache_tmp['sum_y2'], \
                                cache_tmp['n_points'], param, cache_tmp, settings)
    cache_tmp['loglik'] = op_tmp
    cache_tmp['param_n'] = param_tmp
    # pre-compute stuff
    cache = {}
    cache['nn_prior_term'] = float(0.5 * math.log(param.mu_prec) - 0.5 * param.mu_prec * param.mu_mean * param.mu_mean)
    cache['half_log_2pi'] = 0.5 * math.log(2 * math.pi)
    cache['range_n_dim'] = range(data['n_dim'])
    cache['range_n_dim_shuffle'] = range(data['n_dim'])
    cache['log_n_dim'] = math.log(data['n_dim'])
    feat_val2idx = {}   # maps unique values to idx for feat_score_cumsum
    feat_idx2midpoint = {}   # maps idx of interval to midpoint
    feat_score_cumsum_prior = {}         # cumsum of scores of each interval for prior
    feat_k_log_prior = (-np.log(float(data['n_dim']))) * np.ones(data['n_dim'])         # log prior of k
    for feat_id in cache['range_n_dim']:
        x_tmp = data['x_train'][:, feat_id]
        idx_sort = np.argsort(x_tmp)
        feat_unique_values = np.unique(x_tmp[idx_sort])
        feat_val2idx[feat_id] = {}
        n_unique = len(feat_unique_values)
        for n, x_n in enumerate(feat_unique_values):
            feat_val2idx[feat_id][x_n] = n     # even min value may be looked up
        # first "interval" has width 0 since points to the left of that point are chosen with prob 0
        feat_idx2midpoint[feat_id] = np.zeros(n_unique)
        feat_idx2midpoint[feat_id][1:] = (feat_unique_values[1:] + feat_unique_values[:-1]) / 2.0
        # each interval is represented by its midpoint
        diff_feat_unique_values = np.diff(feat_unique_values)
        log_diff_feat_unique_values_norm = np.log(diff_feat_unique_values) \
                            - np.log(feat_unique_values[-1] - feat_unique_values[0])
        feat_score_prior_tmp = np.zeros(n_unique)
        feat_score_prior_tmp[1:] = diff_feat_unique_values
        feat_score_cumsum_prior[feat_id] = np.cumsum(feat_score_prior_tmp)
        if settings.debug == 1:
            # print 'check if all these numbers are the same:'
            # print n_unique, len(feat_score_cumsum_prior[feat_id])
            assert(n_unique == len(feat_score_cumsum_prior[feat_id]))
        if settings.verbose >= 3:
            print 'x (sorted) =  %s' % (x_tmp[idx_sort])
            print 'y (corresponding to sorted x) = %s' % (data['y_train'][idx_sort])
    cache['feat_val2idx'] = feat_val2idx
    cache['feat_idx2midpoint'] = feat_idx2midpoint
    cache['feat_score_cumsum_prior'] = feat_score_cumsum_prior
    if settings.proposal == 'prior':
        cache['feat_score_cumsum'] = cache['feat_score_cumsum_prior']
    cache['feat_k_log_prior']  = feat_k_log_prior
    # use prob_feat instead of score_feat here; else need to pass sum of scores to log_sis_ratio
    cache['prob_feat'] = np.exp(feat_k_log_prior)
    return (param, cache, cache_tmp)


def update_cache_tmp(cache_tmp, data, param, settings):
    cache_tmp['sum_y'] = float(np.sum(data['y_train']))
    cache_tmp['sum_y2'] = float(np.sum(data['y_train'] ** 2))
    op_tmp, param_tmp = compute_normal_normalizer(cache_tmp['sum_y'], cache_tmp['sum_y2'], \
                                cache_tmp['n_points'], param, cache_tmp, settings)
    cache_tmp['loglik'] = op_tmp
    cache_tmp['param_n'] = param_tmp


def no_valid_split_exists(data, cache, train_ids, settings):
    # faster way to check for existence of valid split than find_valid_dimensions
    op = True
    for feat_id in cache['range_n_dim_shuffle']:
        x_min = np.min(data['x_train'][train_ids, feat_id])
        x_max = np.max(data['x_train'][train_ids, feat_id])
        idx_min = cache['feat_val2idx'][feat_id][x_min]
        idx_max = cache['feat_val2idx'][feat_id][x_max]
        if idx_min != idx_max:
            op = False
            break
    return op


def get_reg_stats(y):
    # y is a list of numbers, get_reg_stats(y) returns stats required for computing regression likelihood
    y_ = np.array(y)
    sum_y = float(np.sum(y_))
    sum_y2 = float(np.sum(pow(y_, 2)))
    n_points = len(y_)
    return (sum_y, sum_y2, n_points)


def compute_gamma_param(min_val, alpha, q, init_val=-1.0):
    # alpha, beta are shape and rate of gamma distribution
    # solves for the equation: gammacdf(min_val, shape=alpha, rate=beta) = 1 - q
    # we find the rate parameter of the gamma distribution such that fraction q of the 
    #   total probability mass is assigned to values > min_val
    if init_val < 0:    
        init_val = alpha / 3.0 / min_val
        # intuition: expected mean of the gamma distribution, alpha/beta = 3 * min_val
    solution = fsolve(lambda beta: gdtrc(beta, alpha, min_val) - q, init_val)
    # gdtrc(rate, shape, min_val) returns the integral from x to infinity of the gamma probability density function
    # gdtrc (and gdtr) takes in parameters in the order (rate, shape, min_val) and not (shape, rate, min_val)
    # source: http://gaezipserver.appspot.com/python/sci/scipy/generated/scipy.special.gdtr.html#scipy.special.gdtr
    try:
        assert(abs(gdtrc(solution, alpha, min_val) - q) < 1e-3)
    except AssertionError:
        print 'Failed to obtain the right solution: beta_init = %s, q = %s, ' \
                'gdtrc(solution, alpha, min_val) = %s' \
                % (init_val, q, gdtrc(solution, alpha, min_val))
        print 'Trying a new initial value for beta'
        # new_init = alpha / min_val / 5
        new_init = max(0.001, init_val * 0.9)       # seems to work for compute_gamma_param(min_val, 3.0, 0.9)
        # very low values of new_init (~0) seem to crash; haven't tested for arbitrary combinations of alpha and q
        solution = compute_gamma_param(min_val, alpha, q, new_init)
    return float(solution)


def test_compute_gamma_param(min_val=0.25, alpha=3, q=0.9):
    beta = compute_gamma_param(min_val, alpha, q)
    x = np.random.gamma(alpha, 1/beta, 1000000)
    assert(np.abs(np.mean(x > min_val) - q) < 0.02) 


def compute_gamma_loglik(x, alpha, beta, log_const=None):
    if log_const is None:
        log_const = alpha * math.log(beta) - gammaln(alpha) 
    log_px = log_const + (alpha - 1) * np.log(x) - beta * x
    return log_px


def compute_normal_loglik(x, mu, prec):
    op = 0.5 * (math.log(prec) - math.log(2 * math.pi) - prec * ((x - mu) ** 2))
    return op


def init_performance_storage(data, settings):
    mcmc_tree_predictions = {}
    n_store = int((settings.n_iterations) / settings.n_run_avg)
    print 'n_store = %s' % n_store
    if settings.save == 1:
        for k_data in settings.perf_dataset_keys:
            k_data_n = 'n_' + k_data
            mcmc_tree_predictions[k_data] = {'accum':{}, 'individual':{}}
            # moving n_store to the first dimension makes assignment of varying dimension easier
            # eg. we would have to use pred_prob[:,:,itr_store] for 'class' and pred_prob[:,itr_store] otherwise
            mcmc_tree_predictions[k_data]['pred_prob'] = \
                    np.zeros((n_store, data[k_data_n]))
            mcmc_tree_predictions[k_data]['pred_mean'] = \
                    np.zeros((n_store, data[k_data_n]))
            if settings.store_every_iteration == 1:
                mcmc_tree_predictions[k_data]['individual']['pred_prob'] = \
                        np.zeros((settings.n_iterations, data[k_data_n]))
                mcmc_tree_predictions[k_data]['individual']['pred_mean'] = \
                        np.zeros((settings.n_iterations, data[k_data_n]))
            mcmc_tree_predictions[k_data]['accum']['pred_prob'] = \
                    np.zeros((data[k_data_n]))
            mcmc_tree_predictions[k_data]['accum']['pred_mean'] = \
                    np.zeros((data[k_data_n]))
        mcmc_tree_predictions['run_avg_stats'] = np.zeros((6, n_store))
        if settings.store_every_iteration == 1:
            mcmc_tree_predictions['individual_perf_stats'] = np.zeros((6, settings.n_iterations))
    return mcmc_tree_predictions


def get_k_data_names(settings, k_data):
    if settings.dataset[:8] == 'friedman':
        k_data_tmp = 'f_' + k_data                  # y has been corrupted with noise in this dataset, f is the ground truth
    else:
        k_data_tmp = 'y_' + k_data + '_orig'
    k_data_n = 'n_' + k_data
    return (k_data_tmp, k_data_n)


def store_every_iteration(mcmc_tree_predictions, data, settings, param, itr, pred_tmp, time_current_itr, time_init_current):
    metrics = {}
    for k_data in settings.perf_dataset_keys:
        k_data_tmp, k_data_n = get_k_data_names(settings, k_data)
        for k_store in settings.perf_store_keys:
            mcmc_tree_predictions[k_data]['individual'][k_store][itr] = pred_tmp[k_data][k_store] 
        metrics[k_data] = compute_metrics_regression(data[k_data_tmp], pred_tmp[k_data]['pred_mean'], \
                pred_tmp[k_data]['pred_prob'])
    mcmc_tree_predictions['individual_perf_stats'][:, itr] = \
            [metrics['train']['mse'], metrics['train']['log_prob'], \
            metrics['test']['mse'], metrics['test']['log_prob'], \
            time_current_itr, time.clock() - time_init_current]
    if itr == 0:
        print 'itr, mse_train, log_prob_train, mse_test, log_prob_test, time_current_itr, time including prediction'
    print '%7d, %s' % (itr, mcmc_tree_predictions['individual_perf_stats'][:, itr].T)


if __name__ == "__main__":
    test_compute_gamma_param()
    test_compute_metrics_regression()
    test_compute_nn_normalizer()
