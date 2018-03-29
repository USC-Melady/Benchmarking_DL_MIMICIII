#!/usr/bin/env python
# MCMC for single tree

import numpy as np
from copy import copy
import random
from bart_utils import empty, Tree, softmax, check_if_zero, sample_multinomial, \
        get_parent_id, get_sibling_id, get_depth, get_children_id, stop_split, \
        compute_normal_normalizer, compute_left_right_statistics, precompute, no_valid_split_exists
from pg import Particle, init_run_smc, init_particles

STEP_NAMES = ['grow', 'prune', 'change', 'swap']

class PMCMC(object):
    def __init__(self, data, settings, param, cache, cache_tmp):
        if settings.init_pg == 'smc':
            particles, log_pd, log_weights = init_mcmc_using_smc(data, settings, param, cache, cache_tmp)
        elif settings.init_pg == 'empty':
            n_particles_backup = copy(settings.n_particles)
            settings.n_particles = 1
            particles, log_weights = init_particles(data, settings, param, cache_tmp)
            settings.n_particles = n_particles_backup
            log_pd = -np.inf
            for p in particles:
                p.nodes_processed_itr = [[0]]
                p.grow_nodes_itr = [[0]]
                if settings.verbose >= 2:
                    print('leaf = %s, non_leaf = %s, node_info = %s' % (p.leaf_nodes, p.non_leaf_nodes, p.node_info))
        if settings.verbose >= 2:
            print('\ninitializing particle mcmc')
            print(log_pd)
            print(log_weights)
        self.update_p(particles, log_weights, log_pd, settings)

    def update_p(self, particles, log_weights, log_pd, settings):
        node_info_old = {}
        first_iter = False
        if settings.verbose >= 2:
            print('log_weights = %s' % log_weights)
        k = sample_multinomial(softmax(log_weights))
        try:
            node_info_old = self.p.node_info 
        except AttributeError:
            first_iter = True
            # first iteration probably: self.p would not be present
            pass
        same_tree = node_info_old == particles[k].node_info
        self.p = particles[k] 
        self.log_pd = log_pd
        if settings.verbose >= 2:
            print('pid_sampled = %s' % k)
            print('new tree:')
            self.p.print_tree()
        if not same_tree and settings.verbose >=1:
            print('non-identical trees')
        if k == 0 and not first_iter:
            assert same_tree
        elif same_tree and not first_iter:  # particles from pmcmc during init might be different
            if settings.verbose >= 1:
                print('identical tree without k == 0')
            try:
                check_if_zero(log_weights[k] - log_weights[0])
            except AssertionError:
                print('node_info_old = %s' % node_info_old)
                print('same_tree = %s' % same_tree)
                print('k = %s, particles[k].node_info = %s' % (k, particles[k].node_info))
                print('log_weights[0] = %s, log_weights[k] = %s' % \
                        (log_weights[0], log_weights[k]))
                if not first_iter:
                    print(p_old.log_sis_ratio_d)
                print(particles[0].log_sis_ratio_d)
                print(particles[k].log_sis_ratio_d)
                raise AssertionError
        self.p.check_depth()
        if settings.verbose >= 2:
            print('sampled particle = %5d, ancestry = %s' % (k, self.p.ancestry))
        return not same_tree

    def sample(self, data, settings, param, cache, cache_tmp):
        # Particle Gibbs (PG) sampler
        particles, log_pd, log_weights = init_run_smc(data, settings, param, cache, cache_tmp, self.p)
        change = self.update_p(particles, log_weights, log_pd, settings)
        return change


def init_mcmc_using_smc(data, settings, param, cache, cache_tmp):
    # initialization via PMCMC done using different ESS and 100 particles
    ess_threshold_backup = settings.ess_threshold
    #settings.ess_threshold = 0.1   # FIXME: use the same ESS threshold as before
    n_particles_backup = settings.n_particles
    settings.n_particles = 100
    particles, log_pd, log_weights = init_run_smc(data, settings, param, cache, cache_tmp)
    settings.ess_threshold = ess_threshold_backup
    settings.n_particles = n_particles_backup
    return (particles, log_pd, log_weights)


def sample_tree(data, settings, param, cache, cache_tmp):
    p = TreeMCMC(list(range(data['n_train'])), param, settings, cache_tmp)
    grow_nodes = [0]
    while grow_nodes:
        node_id = grow_nodes.pop(0)
        p.depth = max(p.depth, get_depth(node_id))
        log_psplit = np.log(p.compute_psplit(node_id, param))
        train_ids = p.train_ids[node_id]
        (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
            train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) \
            = p.precomputed_proposal(data, param, settings, cache, node_id, train_ids, log_psplit)
        if do_not_split_node_id:
            p.do_not_split[node_id] = True
        else:
            p.update_left_right_statistics(cache_tmp, node_id, logprior_nodeid, train_ids_left,\
                train_ids_right, loglik_left, loglik_right, feat_id_chosen, split_chosen, \
                idx_split_global, settings, param, data, cache)
            left, right = get_children_id(node_id)
            grow_nodes.append(left)
            grow_nodes.append(right)
            # create mcmc structures
            p.both_children_terminal.append(node_id)
            parent = get_parent_id(node_id) 
            if (node_id != 0) and (parent in p.non_leaf_nodes):
                p.inner_pc_pairs.append((parent, node_id))
            if node_id != 0:
                try:
                    p.both_children_terminal.remove(parent)
                except ValueError:
                    pass
    if settings.debug:
        print('sampled new tree:')
        p.print_tree()
    return p


class TreeMCMC(Tree):
    def __init__(self, train_ids=np.arange(0, dtype='int'), param=empty(), settings=empty(), cache_tmp={}):
        Tree.__init__(self, train_ids, param, settings, cache_tmp)
        self.inner_pc_pairs = []       # list of nodes where both parent/child are non-terminal
        self.both_children_terminal = []

    def compute_log_acc_g(self, node_id, param, len_both_children_terminal, loglik, \
            train_ids_left, train_ids_right, cache, settings, data, grow_nodes):
        # effect of do_not_split does not matter for node_id since it has children
        logprior_children = 0.0
        left, right = get_children_id(node_id)
        if not no_valid_split_exists(data, cache, train_ids_left, settings):
            logprior_children += np.log(self.compute_pnosplit(left, param))
        if not no_valid_split_exists(data, cache, train_ids_right, settings):
            logprior_children += np.log(self.compute_pnosplit(right, param))
        log_acc_prior = np.log(self.compute_psplit(node_id, param)) \
                -np.log(self.compute_pnosplit(node_id, param)) \
            -np.log(len_both_children_terminal) + np.log(len(grow_nodes)) \
            + logprior_children 
        log_acc_loglik = (loglik - self.loglik[node_id])
        log_acc = log_acc_prior + log_acc_loglik
        if settings.verbose >= 2:
            print('compute_log_acc_g: log_acc_loglik = %s, log_acc_prior = %s' \
                    % (log_acc_loglik, log_acc_prior))
        if loglik == -np.inf:   # just need to ensure that an invalid split is not grown
            log_acc = -np.inf
        return log_acc

    def compute_log_inv_acc_p(self, node_id, param, len_both_children_terminal, loglik, grow_nodes, \
            cache, settings, data):
        # 1/acc for PRUNE is acc for GROW except for corrections to both_children_terminal 
        #       and grow_nodes list
        logprior_children = 0.0
        left, right = get_children_id(node_id)
        if not no_valid_split_exists(data, cache, self.train_ids[left], settings):
            logprior_children += np.log(self.compute_pnosplit(left, param))
        if not no_valid_split_exists(data, cache, self.train_ids[right], settings):
            logprior_children += np.log(self.compute_pnosplit(right, param))
        try:
            check_if_zero(logprior_children - self.logprior[left] - self.logprior[right])
        except AssertionError:
            print('oh oh ... looks like a bug in compute_log_inv_acc_p')
            print('term 1 = %s' % logprior_children)
            print('term 2 = %s, 2a = %s, 2b = %s' % (self.logprior[left]+self.logprior[right], \
                     self.logprior[left], self.logprior[right]))
            print('node_id = %s, left = %s, right = %s, logprior = %s' % (node_id, left, right, self.logprior))
            raise AssertionError
        log_inv_acc_prior = np.log(self.compute_psplit(node_id, param)) \
                - np.log(self.compute_pnosplit(node_id, param)) \
                -np.log(len_both_children_terminal) + np.log(len(grow_nodes)) \
                + logprior_children 
        log_inv_acc_loglik = (loglik - self.loglik[node_id])
        log_inv_acc = log_inv_acc_loglik + log_inv_acc_prior
        if settings.verbose >= 2:
            print('compute_log_inv_acc_p: log_acc_loglik = %s, log_acc_prior = %s' \
                    % (-log_inv_acc_loglik, -log_inv_acc_prior))
        assert(log_inv_acc > -np.inf)
        return log_inv_acc

    def sample(self, data, settings, param, cache):
        if settings.mcmc_type == 'growprune':
            step_id = random.randint(0, 1)  # only grow and prune moves permitted
        elif settings.mcmc_type == 'cgm':
            step_id = random.randint(0, 3)  # all 4 moves equally likely (or think of 50% grow/prune, 25% change, 25% swap)
        else:
            raise Exception('invalid mcmc_type')
        log_acc = -np.inf
        log_r = 0
        self.grow_nodes = [n_id for n_id in self.leaf_nodes \
                    if not stop_split(self.train_ids[n_id], settings, data, cache)]
        grow_nodes = self.grow_nodes
        if step_id == 0:        # GROW step
            if not grow_nodes:
                change = False
            else:
                node_id = random.choice(grow_nodes)
                if settings.verbose >= 1:
                    print('grow_nodes = %s, chosen node_id = %s' % (grow_nodes, node_id))
                do_not_split_node_id, feat_id, split, idx_split_global, logprior_nodeid = \
                        self.sample_split_prior(data, param, settings, cache, node_id)
                assert not do_not_split_node_id
                if settings.verbose >= 1:
                    print('grow: do_not_split = %s, feat_id = %s, split = %s' \
                            % (do_not_split_node_id, feat_id, split))
                train_ids = self.train_ids[node_id]
                (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                    compute_left_right_statistics(data, param, cache, train_ids, \
                        feat_id, split, settings)
                loglik = loglik_left + loglik_right
                len_both_children_terminal_new = len(self.both_children_terminal)
                if get_sibling_id(node_id) not in self.leaf_nodes:
                    len_both_children_terminal_new += 1
                log_acc = self.compute_log_acc_g(node_id, param, len_both_children_terminal_new, \
                            loglik, train_ids_left, train_ids_right, cache, settings, data, grow_nodes)
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.update_left_right_statistics(cache_tmp, node_id, logprior_nodeid, \
                            train_ids_left, train_ids_right, loglik_left, loglik_right, \
                            feat_id, split, idx_split_global, settings, param, data, cache)
                    # MCMC specific data structure updates
                    self.both_children_terminal.append(node_id)
                    parent = get_parent_id(node_id) 
                    if (node_id != 0) and (parent in self.non_leaf_nodes):
                        self.inner_pc_pairs.append((parent, node_id))
                    sibling = get_sibling_id(node_id)
                    if sibling in self.leaf_nodes:
                        self.both_children_terminal.remove(parent)
                    change = True
                else:
                    change = False
        elif step_id == 1:      # PRUNE step
            if not self.both_children_terminal:
                change = False      # nothing to prune here
            else:
                node_id = random.choice(self.both_children_terminal)
                feat_id = self.node_info[node_id][0]
                if settings.verbose >= 1:
                    print('prune: node_id = %s, feat_id = %s' % (node_id, feat_id))
                left, right = get_children_id(node_id)
                loglik = self.loglik[left] + self.loglik[right]
                len_both_children_new = len(self.both_children_terminal)
                grow_nodes_tmp = grow_nodes[:]
                grow_nodes_tmp.append(node_id)
                try:
                    grow_nodes_tmp.remove(left)
                except ValueError:
                    pass
                try:
                    grow_nodes_tmp.remove(right)
                except ValueError:
                    pass
                log_acc = - self.compute_log_inv_acc_p(node_id, param, len_both_children_new, \
                                loglik, grow_nodes_tmp, cache, settings, data)
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.remove_leaf_node_statistics(left, settings)
                    self.remove_leaf_node_statistics(right, settings)
                    self.leaf_nodes.append(node_id)
                    self.non_leaf_nodes.remove(node_id)
                    self.logprior[node_id] = np.log(self.compute_pnosplit(node_id, param))
                    # OK to set logprior as above since we know that a valid split exists
                    # MCMC specific data structure updates
                    self.both_children_terminal.remove(node_id)
                    parent = get_parent_id(node_id) 
                    if (node_id != 0) and (parent in self.non_leaf_nodes):
                        self.inner_pc_pairs.remove((parent, node_id))
                    if node_id != 0:
                        sibling = get_sibling_id(node_id) 
                        if sibling in self.leaf_nodes:
                            if settings.debug == 1:
                                assert(parent not in self.both_children_terminal)
                            self.both_children_terminal.append(parent)
                    change = True
                else:
                    change = False
        elif step_id == 2:      # CHANGE
            if not self.non_leaf_nodes:
                change = False
            else:
                node_id = random.choice(self.non_leaf_nodes)
                do_not_split_node_id, feat_id, split, idx_split_global, logprior_nodeid = \
                        self.sample_split_prior(data, param, settings, cache, node_id)
                if settings.verbose >= 1:
                    print('change: node_id = %s, do_not_split = %s, feat_id = %s, split = %s' \
                            % (node_id, do_not_split_node_id, feat_id, split))
                # Note: this just samples a split criterion, not guaranteed to "change" 
                assert(not do_not_split_node_id)
                nodes_subtree = self.get_nodes_subtree(node_id)
                nodes_not_in_subtree = self.get_nodes_not_in_subtree(node_id)
                if settings.debug == 1:
                    set1 = set(list(nodes_subtree) + list(nodes_not_in_subtree))
                    set2 = set(self.leaf_nodes + self.non_leaf_nodes)
                    assert(sorted(set1) == sorted(set2))
                self.create_new_statistics(nodes_subtree, nodes_not_in_subtree, node_id, settings)
                self.node_info_new[node_id] = (feat_id, split, idx_split_global)         
                self.evaluate_new_subtree(data, node_id, param, nodes_subtree, cache, settings)
                # log_acc will be be modified below
                log_acc_tmp, loglik_diff, logprior_diff = self.compute_log_acc_cs(nodes_subtree, node_id)
                if settings.debug == 1:
                    self.check_if_same(log_acc_tmp, loglik_diff, logprior_diff)
                log_acc = log_acc_tmp + self.logprior[node_id] - self.logprior_new[node_id]
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.node_info[node_id] = copy(self.node_info_new[node_id])
                    self.update_subtree(node_id, nodes_subtree, settings)
                    change = True
                else:
                    change = False
        elif step_id == 3:      # SWAP
            if not self.inner_pc_pairs:
                change = False 
            else:
                node_id, child_id = random.choice(self.inner_pc_pairs)
                nodes_subtree = self.get_nodes_subtree(node_id)
                nodes_not_in_subtree = self.get_nodes_not_in_subtree(node_id)
                if settings.debug == 1:
                    set1 = set(list(nodes_subtree) + list(nodes_not_in_subtree))
                    set2 = set(self.leaf_nodes + self.non_leaf_nodes)
                    assert(sorted(set1) == sorted(set2))
                self.create_new_statistics(nodes_subtree, nodes_not_in_subtree, node_id, settings)
                self.node_info_new[node_id] = copy(self.node_info[child_id])
                self.node_info_new[child_id] = copy(self.node_info[node_id])
                if settings.verbose >= 1:
                    print('swap: node_id = %s, child_id = %s' % (node_id, child_id))
                    print('node_info[node_id] = %s, node_info[child_id] = %s' \
                            % (self.node_info[node_id], self.node_info[child_id]))
                self.evaluate_new_subtree(data, node_id, param, nodes_subtree, cache, settings)
                log_acc, loglik_diff, logprior_diff = self.compute_log_acc_cs(nodes_subtree, node_id)
                if settings.debug == 1:
                    self.check_if_same(log_acc, loglik_diff, logprior_diff)
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.node_info[node_id] = copy(self.node_info_new[node_id])
                    self.node_info[child_id] = copy(self.node_info_new[child_id])
                    self.update_subtree(node_id, nodes_subtree, settings)
                    change = True
                else:
                    change = False
        if settings.verbose >= 1:
            print('trying move: step_id = %d, move = %s, log_acc = %s, log_r = %s' \
                    % (step_id, STEP_NAMES[step_id], log_acc, log_r))
        if change:
            self.depth = max([get_depth(node_id) for node_id in \
                    self.leaf_nodes])
            self.loglik_current = sum([self.loglik[node_id] for node_id in \
                    self.leaf_nodes])
            if settings.verbose >= 1:
                print('accepted move: step_id = %d, move = %s' % (step_id, STEP_NAMES[step_id]))
                self.print_stuff()
        if settings.debug == 1:
            both_children_terminal, inner_pc_pairs = self.recompute_mcmc_data_structures()
            print('\nstats from recompute_mcmc_data_structures')
            print('both_children_terminal = %s' % both_children_terminal)
            print('inner_pc_pairs = %s' % inner_pc_pairs)
            assert(sorted(both_children_terminal) == sorted(self.both_children_terminal))
            assert(sorted(inner_pc_pairs) == sorted(self.inner_pc_pairs))
            grow_nodes_new = [n_id for n_id in self.leaf_nodes \
                    if not stop_split(self.train_ids[n_id], settings, data, cache)]
            if change and (step_id == 1):
                print('grow_nodes_new = %s, grow_nodes_tmp = %s' % (sorted(grow_nodes_new), sorted(grow_nodes_tmp)))
                assert(sorted(grow_nodes_new) == sorted(grow_nodes_tmp))
        return (change, step_id)

    def check_if_same(self, log_acc, loglik_diff, logprior_diff):
        # change/swap operations should depend only on what happens in current subtree
        loglik_diff_2 =  sum([self.loglik_new[node] for node in self.leaf_nodes]) \
                        - sum([self.loglik[node] for node in self.leaf_nodes])
        logprior_diff_2 = sum([self.logprior_new[node] for node in self.logprior_new]) \
                         - sum([self.logprior[node] for node in self.logprior])
        log_acc_2 = loglik_diff_2 + logprior_diff_2
        try:
            check_if_zero(log_acc - log_acc_2)
        except AssertionError:
            if not ((log_acc == -np.inf) and (log_acc_2 == -np.inf)):
                print('check if terms match:')
                print('loglik_diff = %s, loglik_diff_2 = %s' % (loglik_diff, loglik_diff_2))
                print('logprior_diff = %s, logprior_diff_2 = %s' % (logprior_diff, logprior_diff_2))
                raise AssertionError

    def compute_log_acc_cs(self, nodes_subtree, node_id):
        # for change or swap operations
        loglik_old = sum([self.loglik[node] for node in nodes_subtree if node in self.leaf_nodes])
        loglik_new = sum([self.loglik_new[node] for node in nodes_subtree if node in self.leaf_nodes])
        loglik_diff = loglik_new - loglik_old
        logprior_old = sum([self.logprior[node] for node in nodes_subtree])
        logprior_new = sum([self.logprior_new[node] for node in nodes_subtree])
        logprior_diff = logprior_new - logprior_old
        log_acc = loglik_diff + logprior_diff
        return (log_acc, loglik_diff, logprior_diff)

    def create_new_statistics(self, nodes_subtree, nodes_not_in_subtree, node_id, settings):
        self.node_info_new = self.node_info.copy()
        self.train_ids_new = {}
        self.loglik_new = {}
        self.logprior_new = {}
        self.sum_y_new = {}
        self.sum_y2_new = {}
        self.n_points_new = {}
        self.param_n_new = {}
        for node in nodes_not_in_subtree:
            self.loglik_new[node] = self.loglik[node]
            self.logprior_new[node] = self.logprior[node]
            self.train_ids_new[node] = self.train_ids[node]
            self.sum_y_new[node] = self.sum_y[node]
            self.sum_y2_new[node] = self.sum_y2[node]
            self.n_points_new[node] = self.n_points[node]
            self.param_n_new[node] = self.param_n[node]
        for node in nodes_subtree:
            self.loglik_new[node] = -np.inf
            self.logprior_new[node] = -np.inf
            self.train_ids_new[node] = np.arange(0, dtype='int')
            self.sum_y_new[node] = 0.
            self.sum_y2_new[node] = 0.
            self.n_points_new[node] = 0
            self.param_n_new[node] = self.param_n[node][:]

    def evaluate_new_subtree(self, data, node_id_start, param, nodes_subtree, cache, settings):
        for i in self.train_ids[node_id_start]:
            x_, y_ = data['x_train'][i, :], data['y_train'][i]
            node_id = copy(node_id_start)
            while True:
                self.sum_y_new[node_id] += y_
                self.sum_y2_new[node_id] += y_ ** 2
                self.n_points_new[node_id] += 1
                self.train_ids_new[node_id] = np.append(self.train_ids_new[node_id], i)
                if node_id in self.leaf_nodes:
                    break
                left, right = get_children_id(node_id)
                feat_id, split, idx_split_global = self.node_info_new[node_id]   # splitting on new criteria
                if x_[feat_id] <= split:
                    node_id = left
                else:
                    node_id = right
        for node_id in nodes_subtree:
            self.loglik_new[node_id] = -np.inf
            if self.n_points_new[node_id] > 0:
                self.loglik_new[node_id], self.param_n_new[node_id] = \
                        compute_normal_normalizer(self.sum_y_new[node_id], self.sum_y2_new[node_id], \
                                self.n_points_new[node_id], param, cache, settings)
            if node_id in self.leaf_nodes:
                if stop_split(self.train_ids_new[node_id], settings, data, cache):
                # if leaf is empty, logprior_new[node_id] = 0.0 is incorrect; however
                #      loglik_new[node_id] = -np.inf will reject move to a tree with empty leaves
                    self.logprior_new[node_id] = 0.0
                else:
                    # node with just 1 data point earlier could have more data points now 
                    self.logprior_new[node_id] = np.log(self.compute_pnosplit(node_id, param))
            else:
                # split probability might have changed if train_ids have changed
                self.recompute_prob_split(data, param, settings, cache, node_id)
        if settings.debug == 1:
            try:
                check_if_zero(self.loglik[node_id_start] - self.loglik_new[node_id_start])
            except AssertionError:
                print('train_ids[node_id_start] = %s, train_ids_new[node_id_start] = %s' \
                        % (self.train_ids[node_id_start], self.train_ids_new[node_id_start]))
                raise AssertionError
    
    def update_subtree(self, node_id, nodes_subtree, settings):
        for node in nodes_subtree:
            self.loglik[node] = copy(self.loglik_new[node])
            self.logprior[node] = copy(self.logprior_new[node])
            self.train_ids[node] = self.train_ids_new[node].copy()
            self.sum_y[node] = copy(self.sum_y_new[node])
            self.sum_y2[node] = copy(self.sum_y2_new[node])
            self.n_points[node] = copy(self.n_points_new[node])
            self.param_n[node] = self.param_n_new[node][:]

    def print_stuff(self):
        print('tree statistics:')
        print('leaf nodes = ')
        print(self.leaf_nodes) 
        print('non leaf nodes = ')
        print(self.non_leaf_nodes) 
        print('inner pc pairs')
        print(self.inner_pc_pairs) 
        print('both children terminal')
        print(self.both_children_terminal)
        print('loglik = ')
        print(self.loglik)
        print('logprior = \n%s' % self.logprior)
        print('do_not_split = \n%s'  % self.do_not_split)
        print() 

    def get_nodes_not_in_subtree(self, node_id):
        all_nodes = set(self.leaf_nodes + self.non_leaf_nodes)
        reqd_nodes = all_nodes - set(self.get_nodes_subtree(node_id))
        return list(reqd_nodes)

    def get_nodes_subtree(self, node_id):
        # NOTE: current node_id is included in nodes_subtree as well
        node_list = []
        expand = [node_id]
        while len(expand) > 0:
            node = expand.pop(0) 
            node_list.append(node)
            if node not in self.leaf_nodes:
                left, right = get_children_id(node)
                expand.append(left)
                expand.append(right)
        return node_list

    def recompute_mcmc_data_structures(self):   #, settings, param):
        nodes_to_visit = sorted(set(self.leaf_nodes + self.non_leaf_nodes))
        both_children_terminal = []
        inner_pc_pairs = []
        while nodes_to_visit:
            node_id = nodes_to_visit[0]
            parent = get_parent_id(node_id)
            if (node_id != 0) and (node_id in self.non_leaf_nodes) and (parent in self.non_leaf_nodes):
                inner_pc_pairs.append((parent, node_id))
            if node_id != 0:
                sibling = get_sibling_id(node_id)
                if (node_id in self.leaf_nodes) and (sibling in self.leaf_nodes) \
                        and (parent not in both_children_terminal):
                    both_children_terminal.append(parent)
            nodes_to_visit.remove(node_id)
        return (both_children_terminal, inner_pc_pairs)


def init_tree_mcmc(data, settings, param, cache, cache_tmp):
    pmcmc = None
    if settings.mcmc_type == 'cgm' or settings.mcmc_type == 'growprune':
        # Chipman et al's mcmc version
        if settings.init_mcmc == 'random':
            # initialize with a random tree
            p = sample_tree(data, settings, param, cache, cache_tmp)
        else:
            # initialize with empty tree (works better in my experience since MCMC takes a while to "unlearn")
            p = TreeMCMC(np.arange(data['n_train'], dtype='int'), param, settings, cache_tmp)
        if settings.verbose >= 2:
            print('*'*80)
            print('initial tree:')
            p.print_stuff()
            print('*'*80)
    elif settings.mcmc_type == 'pg':
        # Particle-MCMC
        # init_pmcmc handled within PMCMC class definition 
        pmcmc = PMCMC(data, settings, param, cache, cache_tmp)
        p = pmcmc.p
    else:
        raise Exception
    pred_tmp = {}
    return (p, pred_tmp, pmcmc)


def run_mcmc_single_tree(p, settings, data, param, cache, change, mcmc_counts, cache_tmp, pmcmc=None):
    if settings.mcmc_type == 'cgm' or settings.mcmc_type == 'growprune':
        (change, step_id) = p.sample(data, settings, param, cache)
    elif settings.mcmc_type == 'pg':
        change = pmcmc.sample(data, settings, param, cache, cache_tmp)
        p = pmcmc.p
        if settings.debug == 1:
            print('change = %s, p = %s, pmcmc.p = %s' % (str(change), p, pmcmc.p))
            print('current logprob = %f' % pmcmc.p.compute_logprob())
            print()
    return (p, change)
