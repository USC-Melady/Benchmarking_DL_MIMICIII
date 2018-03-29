#!/usr/bin/env python
# "Bayesian Backfitting" MCMC algorithm for learning additive trees
#
# Example usage: ./bart.py  --alpha_s 0.95 --beta_s 0.5 --k_bart 2 --dataset friedman-5 --m_bart 200 --n_iter 1000 --save 0 -v 0
#
# high level code:
# for i in range(n_iterations):
#     sample lambda_bart_i | T_{*, i-1}, M_{*, i-1} which is just a draw from gamma (or inverse gamma)
#     for j in trees:
#         sample T_{j,i} | (T_{j,i-1}, T_{1:j-1, i}, T_{j+1:J, i-1}, M_{1:j-1, i}, M_{j+1:J, i-1})    # M_{j, i} integrated out
#         - get a single R_j that summarizes the current residual and use this as target
#         - sample T_{j,i} using PG, CGM or GrowPrune
#         - PG: run conditional SMC and draw a single tree sample from the posterior 
#         - CGM: MCMC used in BART paper by Chipman et al.
#         - GrowPrune: use just grow and prune proposals
#         sample M_{j,i} | (T_{j,i}, T_{1:j-1, i}, T_{j+1:J, i-1}, M_{1:j-1, i}, M_{j+1:J, i-1}) => sample M_{j,i} | T_{j,i} (due to independence structure)
#         compute predictions of T_{j, i} on both 
#    aggregate preidctions of T_{*, i} for computing final predictions
#
# see paper for further details

import sys
import optparse
import math
import pickle as pickle
import numpy as np
import pprint as pp
from scipy.special import gammaln, digamma
from copy import copy
from bart_utils import Tree, process_command_line, get_filename_bart, \
        compute_metrics_regression, precompute, update_cache_tmp, \
        compute_gamma_loglik, compute_normal_loglik, init_performance_storage, \
        get_k_data_names, store_every_iteration, load_data
import random
from pg import Particle
from treemcmc import TreeMCMC, init_tree_mcmc, run_mcmc_single_tree
import time
# setting numpy options to debug RuntimeWarnings
#np.seterr(divide='raise')
np.seterr(divide='ignore')      # to avoid warnings for np.log(0)
np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200)
np.set_printoptions(threshold=np.inf)  # to print full numpy array


class BART(object):
    def __init__(self, data, param, settings, cache, cache_tmp):
        self.trees = []
        self.pmcmc_objects = []
        self.pred_val_mat_train = np.zeros((data['n_train'], settings.m_bart))
        self.update_pred_val_sum()      # updates pred_val_sum_train
        for i_t in range(settings.m_bart):
            p, pred_tmp, pmcmc = init_tree_mcmc(data, settings, param, cache, cache_tmp)
            sample_param(p, settings, param, False)  #NOTE: deterministic initialization if True
            self.trees.append(p)
            self.pmcmc_objects.append(pmcmc)
            self.update_pred_val(i_t, data, param, settings)
        self.lambda_logprior = compute_gamma_loglik(param.lambda_bart, param.alpha_bart, param.beta_bart)

    def update_pred_val(self, i_t, data, param, settings):
        self.trees[i_t].gen_rules_tree()
        tmp = self.trees[i_t].predict_real_val_fast(data['x_train'], param, settings)
        if settings.debug == 1:
            print('current data[y_train] = %s' % data['y_train'])
            print('predictions of current tree = %s' % tmp)
        self.pred_val_sum_train += tmp
        self.pred_val_sum_train -= self.pred_val_mat_train[:, i_t]
        # could use residual here for minor performance speedup
        self.pred_val_mat_train[:, i_t] = tmp
        if settings.debug == 1:
            print('current data[y_train_orig] = %s' % data['y_train_orig'])
            print('predictions of current bart = %s' % self.pred_val_sum_train)

    def update_pred_val_sum(self):
        self.pred_val_sum_train = self.pred_val_mat_train.sum(1)
        
    def compute_mse_within_without(self, i_t, data, settings):
        pred_tree = self.pred_val_mat_train[:, i_t]
        var_pred_tree = np.var(pred_tree)
        pred_without_tree = self.pred_val_sum_train - self.pred_val_mat_train[:, i_t]
        mse_tree = compute_mse(data['y_train_orig'], pred_tree)
        mse_without_tree = compute_mse(data['y_train_orig'], pred_without_tree)
        return (mse_tree, mse_without_tree, var_pred_tree)

    def update_residual(self, i_t, data):
        residual = data['y_train_orig'] - self.pred_val_sum_train + self.pred_val_mat_train[:, i_t]
        data['y_train'] = residual

    def sample_labels(self, data, settings, param):
        data['y_train_orig'] = self.pred_val_sum_train + np.random.randn(data['n_train']) / math.sqrt(param.lambda_bart)

    def predict(self, x, y, param, settings):
        pred_prob = np.zeros(x.shape[0])
        pred_val = np.zeros(x.shape[0])
        log_const = 0.5 * math.log(param.lambda_bart) - 0.5 * math.log(2 * math.pi)
        for i_t in range(settings.m_bart):
            exec(self.trees[i_t].rules)    # apply rules to "x" and create "leaf_id"
            pred_val += self.trees[i_t].pred_val_n[leaf_id]
        pred_prob = np.exp(- 0.5 * param.lambda_bart * ((y - pred_val) ** 2) + log_const)
        d = {'pred_mean': pred_val, 'pred_prob': pred_prob}
        return d
    
    def predict_train(self, data, param, settings):
        pred_val = self.pred_val_sum_train.copy()
        log_const = 0.5 * math.log(param.lambda_bart) - 0.5 * math.log(2 * math.pi)
        loglik = - 0.5 * param.lambda_bart * ((data['y_train_orig'] - pred_val) ** 2) + log_const
        pred_prob = np.exp(loglik)
        d = {'pred_mean': pred_val, 'pred_prob': pred_prob}
        return d

    def sample_lambda_bart(self, param, data, settings):
        lambda_alpha = param.alpha_bart + 0.5 * data['n_train']
        lambda_beta = param.beta_bart + 0.5 * np.sum((data['y_train_orig'] - self.pred_val_sum_train) ** 2)
        param.lambda_bart = float(np.random.gamma(lambda_alpha, 1.0 / lambda_beta, 1))
        param.log_lambda_bart = math.log(param.lambda_bart)
        self.lambda_logprior = compute_gamma_loglik(param.lambda_bart, param.alpha_bart, param.beta_bart)

    def compute_train_mse(self, data, settings):
        """
        NOTE: inappropriate if y_train_orig contains noise
        """
        mse_train = compute_mse(data['y_train_orig'], self.pred_val_sum_train)
        if settings.verbose >= 1:
            print('mse train = %.3f' % (mse_train))
        return mse_train

    def compute_train_mse_orig(self, data, settings):
        mse_train_orig = compute_mse(data['f_train'], self.pred_val_sum_train)
        return mse_train_orig

    def compute_train_loglik(self, data, settings, param):
        """
        NOTE: inappropriate if y_train_orig contains noise
        """
        mse_train = compute_mse(data['y_train_orig'], self.pred_val_sum_train)
        loglik_train = 0.5 * data['n_train'] * (math.log(param.lambda_bart) \
                        - math.log(2 * math.pi) - param.lambda_bart * mse_train)
        return (loglik_train, mse_train)


def compute_mse(true, pred):
    return np.mean((true - pred) ** 2)


def sample_param(p, settings, param, set_to_mean=False):
    """
    prediction at node (draw from posterior over leaf parameter); 
    Note that CART prediction uses posterior mean
    """
    p.pred_val_n = np.inf * np.ones(max(p.leaf_nodes)+1)
    p.pred_val_logprior = 0.
    for node_id in p.leaf_nodes:
        # NOTE: only for optype == 'real' and prior == 'bart'
        mu_mean_post, mu_prec_post = p.param_n[node_id]
        if set_to_mean:
            p.pred_val_n[node_id] = 0. + mu_mean_post
        else:
            p.pred_val_n[node_id] = float(np.random.randn(1) / np.sqrt(mu_prec_post)) + mu_mean_post
        p.pred_val_logprior += compute_normal_loglik(p.pred_val_n[node_id], param.mu_mean, param.mu_prec)


def center_labels(data, settings):
    data['y_train_mean'] = np.mean(data['y_train'])
    data['y_train'] -= data['y_train_mean']
    data['y_test'] -= data['y_train_mean']


def backup_target(data, settings):
    data['y_train_orig'] = data['y_train'].copy()
    data['y_test_orig'] = data['y_test'].copy()


def main():
    settings = process_command_line()
    print('Current settings:')
    pp.pprint(vars(settings))

    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    random.seed(settings.init_id * 1000)

    # load data
    print('Loading data ...')
    data = load_data(settings)
    print('Loading data ... completed')
    if settings.center_y:
        print('center_y = True; centering the y variables at mean(data[y_train])')
        center_labels(data, settings)
    backup_target(data, settings)
   
    #pre-compute & initialize
    time_start = time.clock()
    param, cache, cache_tmp = precompute(data, settings)
    bart = BART(data, param, settings, cache, cache_tmp)
    time_initialization = time.clock() - time_start

    # initialize stuff for results
    mcmc_stats = np.zeros((settings.m_bart, settings.n_iterations, 10))
    mcmc_stats_bart = np.zeros((settings.n_iterations, 10))
    mcmc_stats_bart_desc = ['loglik', 'logprior', 'logprob', \
                            'mean depth', 'mean num_leaves', 'mean num_nonleaves', 'mean change', \
                            'mse_train', 'lambda_bart', 'time_itr']
    mcmc_counts = None
    mcmc_tree_predictions = init_performance_storage(data, settings)
    n_burn_in = 0       
    # NOTE: predictions are stored without discarding burn-in
    assert n_burn_in == 0
    
    time_init = time.clock()
    time_init_run_avg = time.clock()
    itr_run_avg = 0
    change = True
    tree_order = list(range(settings.m_bart))

    print('initial settings:')
    print('lambda_bart value = %.3f' % param.lambda_bart)
    loglik_train, mse_train = bart.compute_train_loglik(data, settings, param)
    print('mse train = %.3f, loglik_train = %.3f' % (mse_train, loglik_train))
    
    for itr in range(settings.n_iterations):
        time_init_current = time.clock()
        if settings.verbose >= 1:
            print('\n%s BART iteration = %7d %s' % ('*'*30, itr, '*'*30))

        logprior = 0.

        if settings.sample_y == 1 and settings.mcmc_type != 'prior':    # Successive conditional simulator
            bart.sample_labels(data, settings, param)

        # sampling lambda_bart
        bart.sample_lambda_bart(param, data, settings)
        time_sample_lambda = time.clock() - time_init_current
        logprior += bart.lambda_logprior

        random.shuffle(tree_order)
        for i_t in tree_order:
            if settings.debug == 1:
                print('\ntree_id = %3d' % i_t)
            time_init_current_tree = time.clock()
            # update data['y_train']
            bart.update_residual(i_t, data)
            update_cache_tmp(cache_tmp, data, param, settings)

            # MCMC for i_t'th tree
            bart.trees[i_t].update_loglik_node_all(data, param, cache, settings)
            (bart.trees[i_t], change) = run_mcmc_single_tree(bart.trees[i_t], settings, data, param, \
                                        cache, change, mcmc_counts, cache_tmp, bart.pmcmc_objects[i_t])

            # update parameters
            sample_param(bart.trees[i_t], settings, param)
            logprior += bart.trees[i_t].pred_val_logprior
           
            # update pred_val
            bart.update_pred_val(i_t, data, param, settings)

            # update stats
            # 'change' indicates whether MCMC move was accepted 
            bart.trees[i_t].update_depth()
            mcmc_stats[i_t, itr, [3,6,7,8,9]] = np.array([bart.trees[i_t].depth, \
                    len(bart.trees[i_t].leaf_nodes), len(bart.trees[i_t].non_leaf_nodes), \
                    change, time.clock() - time_init_current_tree])
            # NOTE: this logprior computation does not affect timing
            if settings.mcmc_type == 'cgm' or settings.mcmc_type == 'growprune':
                mcmc_stats[i_t, itr, 1] = bart.trees[i_t].compute_logprior()
            else:
                mcmc_stats[i_t, itr, 1] = -np.inf
                #NOTE: compute_logprior could be incorrect for PG 
                # (prior over feature_ids is 1/D rather than 1/numValidDimensions)
        
        if settings.sample_y == 1 and settings.mcmc_type == 'prior':    # Marginal conditional simulator
            bart.sample_labels(data, settings, param)

        if settings.mcmc_type == 'cgm' or settings.mcmc_type == 'growprune':
            logprior += float(np.sum(mcmc_stats[:, itr, 1]))    
        else:
            logprior = -np.inf
        loglik_train, mse_train = bart.compute_train_loglik(data, settings, param)
        logprob_bart = logprior + loglik_train
#        mcmc_stats_bart_desc = 0: loglik, 1: logprior, 2: logprob, 
#                    3: mean depth, 4: mean num_leaves, 5: mean num_nonleaves, 6: mean change, 
#                    7: mse_train, 8: lambda_bart, 9: time_itr
        mcmc_stats_bart[itr, :3] = [loglik_train, logprior, logprob_bart]
        mcmc_stats_bart[itr, 3:7] = np.mean(mcmc_stats[:, itr, [3,6,7,8]], 0)     # depth, #leaf, #nonleaf, change
        mcmc_stats_bart[itr, -3:-1] = [mse_train, param.lambda_bart]
        mcmc_stats_bart[itr, -1] = np.sum(mcmc_stats[:, itr, -1]) + time_sample_lambda  # total time per iteration
        if itr == 0:
            mcmc_stats_bart[itr, -1] += time_initialization
        if settings.verbose >=2 :
            print('fraction of trees in which MCMC move was accepted = %.3f' % mcmc_stats_bart[itr, 6])
        if (settings.save == 1):
            for tree in bart.trees:
                tree.gen_rules_tree()
            pred_tmp = {'train': bart.predict_train(data, param, settings), \
                        'test': bart.predict(data['x_test'], data['y_test_orig'], param, settings)}
            for k_data in settings.perf_dataset_keys:
                for k_store in settings.perf_store_keys:
                    mcmc_tree_predictions[k_data]['accum'][k_store] += pred_tmp[k_data][k_store]
            if itr == 0 and settings.verbose >= 1:
                print('Cumulative: itr, itr_run_avg, [mse train, logprob_train, mse test, ' \
                    'logprob_test, time_mcmc, time_mcmc_prediction], time_mcmc_cumulative')
                print('itr, [mse train, logprob_train, mse test, ' \
                    'logprob_test, time_mcmc, time_mcmc+time_prediction]')
            if settings.store_every_iteration == 1:
                store_every_iteration(mcmc_tree_predictions, data, settings, param, itr, \
                                        pred_tmp, mcmc_stats_bart[itr, -1], time_init_current)
            if (itr > 0) and (itr % settings.n_run_avg == (settings.n_run_avg - 1)):
                metrics = {}
                for k_data in settings.perf_dataset_keys:
                    k_data_tmp, k_data_n = get_k_data_names(settings, k_data)
                    for k_store in settings.perf_store_keys:
                        mcmc_tree_predictions[k_data][k_store][itr_run_avg] = \
                            mcmc_tree_predictions[k_data]['accum'][k_store] / (itr + 1)
                    metrics[k_data] = compute_metrics_regression(data[k_data_tmp], \
                            mcmc_tree_predictions[k_data]['pred_mean'][itr_run_avg], \
                            mcmc_tree_predictions[k_data]['pred_prob'][itr_run_avg])
                itr_range = list(range(itr_run_avg * settings.n_run_avg, (itr_run_avg + 1) * settings.n_run_avg))
                if settings.debug == 1:
                    print('itr_range = %s' % itr_range)
                time_mcmc_train = np.sum(mcmc_stats_bart[itr_range, -1])
                mcmc_tree_predictions['run_avg_stats'][:, itr_run_avg] = \
                        [ metrics['train']['mse'], metrics['train']['log_prob'], \
                        metrics['test']['mse'], metrics['test']['log_prob'], \
                        time_mcmc_train, time.clock() - time_init_run_avg ]
                if settings.verbose >= 1:
                    print('Cumulative: %7d, %7d, %s, %.2f' % \
                        (itr, itr_run_avg, mcmc_tree_predictions['run_avg_stats'][:, itr_run_avg].T, \
                        np.sum(mcmc_tree_predictions['run_avg_stats'][-2, :itr_run_avg+1])))
                itr_run_avg += 1
                time_init_run_avg = time.clock()
    
    # print results
    print('\nTotal time (seconds) = %f' % (time.clock() - time_init))
    if settings.verbose >=2:
        print('mcmc_stats_bart[:, 3:] (not cumulative) = ')
        print('mean depth, mean num_leaves, mean num_nonleaves, ' + \
                'mean change, mse_train, lambda_bart, time_itr')
        print(mcmc_stats_bart[:, 3:])
    if settings.verbose >=1:
        print('mean of mcmc_stats_bart (discarding first 50% of the chain)')
        itr_start = mcmc_stats_bart.shape[0] / 2
        for k, s in enumerate(mcmc_stats_bart_desc):
            print('%20s\t%.2f' % (s, np.mean(mcmc_stats_bart[itr_start:, k])))

    if settings.save == 1:
        print('predictions averaged across all previous additive trees:')
        print('mse train, mean log_prob_train, mse test, mean log_prob_test')
        print(mcmc_tree_predictions['run_avg_stats'][:4,:].T)

    # Write results to disk
    if settings.save == 1:
        filename = get_filename_bart(settings)
        print('filename = ' + filename)
        results = {}
        results['mcmc_stats_bart'] = mcmc_stats_bart
        results['mcmc_stats_bart_desc'] = mcmc_stats_bart_desc
        if settings.store_all_stats:
            results['mcmc_stats'] = mcmc_stats
        results['settings'] = settings
        if settings.dataset[:8] == 'friedman' or settings.dataset[:3] == 'toy':
            results['data'] = data
        pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        filename2 = filename[:-1] + 'tree_predictions.p'
        print('predictions stored in file: %s' % filename2)
        pickle.dump(mcmc_tree_predictions, open(filename2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
