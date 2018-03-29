#!/usr/bin/env python
#

import random
import numpy as np
from copy import copy
from bart_utils import empty, Tree, logsumexp, softmax, check_if_zero, get_children_id
from itertools import izip, count


class Particle(Tree):
    def __init__(self, train_ids=np.arange(0, dtype='int'), param=empty(), settings=empty(), cache_tmp={}):
        Tree.__init__(self, train_ids, param, settings, cache_tmp)
        self.ancestry = []
        self.nodes_processed_itr = []
        self.grow_nodes_itr = []
        self.log_sis_ratio_d = {}
        if cache_tmp:
            self.do_not_grow = False
            self.grow_nodes = [0]

    def process_node_id(self, data, param, settings, cache, node_id):
        if self.do_not_split[node_id]:
            log_sis_ratio = 0.0
        else:
            log_psplit = np.log(self.compute_psplit(node_id, param))
            train_ids = self.train_ids[node_id]
            left, right = get_children_id(node_id)
            if settings.verbose >= 4:
                print 'train_ids for this node = %s' % train_ids
            (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
                train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) \
                = self.prior_proposal(data, param, settings, cache, node_id, train_ids, log_psplit)
            if do_not_split_node_id:
                self.do_not_split[node_id] = True
            else:
                self.update_left_right_statistics(cache_tmp, node_id, logprior_nodeid, train_ids_left,\
                    train_ids_right, loglik_left, loglik_right, feat_id_chosen, split_chosen, \
                    idx_split_global, settings, param, data, cache)
                self.grow_nodes.append(left)
                self.grow_nodes.append(right)
        return (log_sis_ratio)

    def grow_next(self, data, param, settings, cache):
        """ grows just one node at a time (nodewise expansion)
            breaks after processing the first non do_not_grow node or when grow_nodes is empty
            Note that multiple nodes could be killed in a single grow_next call 
        """
        # FIXME: refactor without the do_not_grow option; it made sense for SMC paper, but not for PG
        do_not_grow = True
        log_sis_ratio = 0.0
        nodes_processed = []
        if not self.grow_nodes:
            if settings.verbose >= 2:
                print 'None of the leaves can be grown any further: Current ' \
                    'depth = %3d, Skipping grow_next' % self.depth
        else:
            while True:
                # loop through current leaf nodes, process first "non do_not_grow" node and break; 
                # if none of the nodes can be processed, do_not_grow = True
                remove_position = 0     # just pop the oldest node
                node_id = self.grow_nodes.pop(remove_position)
                nodes_processed.append(node_id)
                do_not_grow = do_not_grow and self.do_not_split[node_id]
                if self.do_not_split[node_id]:
                    if settings.verbose >= 3:
                        print 'Skipping split at node_id %3d' % node_id
                    if not self.grow_nodes:
                        break
                else:
                    log_sis_ratio += self.process_node_id(data, param, settings, cache, node_id)
                    break           # you have processed a non do_not_grow node, take a break!
            self.loglik_current = self.compute_loglik()
        self.log_sis_ratio = log_sis_ratio
        self.do_not_grow = do_not_grow
        if nodes_processed:
            self.nodes_processed_itr.append(nodes_processed)
    
    def check_nodes_processed_itr(self, settings):
        tmp = set([])
        for nodes in self.nodes_processed_itr:
            for node in nodes:
                if node in tmp:
                    print 'node = %s present multiple times in nodes_processed_itr = %s' % \
                            (node, self.nodes_processed_itr)
                    raise Exception
                else:
                    tmp.add(node)
                        

def update_particle_weights(particles, log_weights, settings):
    for n, p in enumerate(particles):
        if settings.verbose >= 2:
            print 'pid = %5d, log_sis_ratio = %f' % (n, p.log_sis_ratio)
        log_weights[n] += p.log_sis_ratio
    weights_norm = softmax(log_weights)     # normalized weights
    ess = 1. / np.sum(weights_norm ** 2) / settings.n_particles
    log_pd = logsumexp(log_weights)
    return (log_pd, ess, log_weights, weights_norm)


def resample(particles, log_weights, settings, log_pd, ess, weights_norm, tree_pg):
    if ess <= settings.ess_threshold:
        if tree_pg:
            pid_list = resample_pids_basic(settings, settings.n_particles-1, weights_norm)
            random.shuffle(pid_list)    # shuffle so that particle is assigned randomly
            pid_list.insert(0, 0)
        else:
            pid_list = resample_pids_basic(settings, settings.n_particles, weights_norm)
        log_weights = np.ones(settings.n_particles) * (log_pd - np.log(settings.n_particles)) 
    else:
        pid_list = range(settings.n_particles)
    if settings.verbose >= 2:
        print 'ess = %s, ess_threshold = %s' % (ess, settings.ess_threshold)
        print 'new particle ids = '
        print pid_list
    op = create_new_particles(particles, pid_list, settings)
    # update ancestry
    for pid, p in izip(pid_list, op):
        p.ancestry.append(pid)
    return (op, log_weights)


def resample_pids_basic(settings, n_particles, prob):
    if settings.resample == 'multinomial':
        pid_list = sample_multinomial_numpy(n_particles, prob)
    elif settings.resample == 'systematic':
        pid_list = systematic_sample(n_particles, prob)
    return pid_list


def sample_multinomial_numpy(n_particles, prob):
        indices = np.random.multinomial(n_particles, prob, size=1)
        pid_list = [pid for pid, cnt in enumerate(indices.flat) \
                    for n in range(cnt)]
        return pid_list


def create_new_particles(particles, pid_list, settings):
    """ particles that occur just once after resampling are not 'copied' """
    list_allocated = set([])
    op = []
    for i, pid in enumerate(pid_list):
        if pid not in list_allocated:
            op.append(particles[pid])
        else:
            op.append(copy_particle(particles[pid], settings))
        list_allocated.add(pid)
    return op


def copy_particle(p, settings): 
    # TODO: lots of unnecessary copying for PG; reduce memory requirement
    op = Particle()
    # lists
    op.leaf_nodes = p.leaf_nodes[:]
    op.non_leaf_nodes = p.non_leaf_nodes[:]
    op.ancestry = p.ancestry[:]
    op.nodes_processed_itr = [x[:] for x in p.nodes_processed_itr]
    op.grow_nodes = p.grow_nodes[:]
    op.grow_nodes_itr = [x[:] for x in p.grow_nodes_itr]
    # dictionaries
    op.do_not_split = p.do_not_split.copy()
    op.log_sis_ratio_d = p.log_sis_ratio_d.copy()
    op.sum_y = p.sum_y.copy()
    op.sum_y2 = p.sum_y2.copy()
    op.n_points = p.n_points.copy()
    op.param_n = p.param_n.copy()
    op.train_ids = p.train_ids.copy()
    op.node_info = p.node_info.copy()
    op.loglik = p.loglik.copy()
    op.logprior = p.logprior.copy()
    # other variables
    op.depth = copy(p.depth)
    op.do_not_grow = copy(p.do_not_grow)
    op.loglik_current = copy(p.loglik_current)
    return op


def systematic_sample(n, prob):
    """ systematic re-sampling algorithm.
    Note: objects with > 1/n probability (better than average) are guaranteed to occur atleast once.
    see section 2.4 of 'Comparison of Resampling Schemes for Particle Filtering' by Douc et. al for more info.
    """
    assert(n == len(prob))
    assert(abs(np.sum(prob) - 1) < 1e-10)
    cum_prob = np.cumsum(prob)
    u = np.random.rand(1) / float(n)
    i = 0
    indices = []
    while True:
        while u > cum_prob[i]:
            i += 1
        indices.append(i)
        u += 1/float(n)
        if u > 1:
            break
    return indices


def init_particles(data, settings, param, cache_tmp):
    particles = [Particle(np.arange(data['n_train']), param, settings, cache_tmp) \
            for n in range(settings.n_particles)]
    log_weights = np.array([p.loglik[0] for p in particles]) - np.log(settings.n_particles)
    return (particles, log_weights)


def grow_next_pg(p, tree_pg, itr, settings):
    p.log_sis_ratio = 0.
    p.do_not_grow = False
    p.grow_nodes = []
    try:
        nodes_processed = tree_pg.nodes_processed_itr[itr]
        p.nodes_processed_itr.append(nodes_processed[:])
        for node_id in nodes_processed[:-1]:
            assert(tree_pg.do_not_split[node_id])
            p.do_not_split[node_id] = True
        node_id = nodes_processed[-1]
        if node_id in tree_pg.node_info:
            left, right = get_children_id(node_id)
            log_sis_ratio_loglik_new = tree_pg.loglik[left] + tree_pg.loglik[right] - tree_pg.loglik[node_id]
            try:
                log_sis_ratio_loglik_old, log_sis_ratio_prior = tree_pg.log_sis_ratio_d[node_id] 
            except KeyError:
                print 'tree_pg: node_info = %s, log_sis_ratio_d = %s' % (tree_pg.node_info, tree_pg.log_sis_ratio_d)
                raise KeyError
            if settings.verbose >= 2:
                print 'log_sis_ratio_loglik_old = %s' % log_sis_ratio_loglik_old
                print 'log_sis_ratio_loglik_new = %s' % log_sis_ratio_loglik_new
            p.log_sis_ratio = log_sis_ratio_loglik_new + log_sis_ratio_prior
            tree_pg.log_sis_ratio_d[node_id] = (log_sis_ratio_loglik_new, log_sis_ratio_prior)
            p.log_sis_ratio_d[node_id] = tree_pg.log_sis_ratio_d[node_id]
            p.non_leaf_nodes.append(node_id)
            try:
                p.leaf_nodes.remove(node_id)
            except ValueError:
                print 'warning: unable to remove node_id = %s from leaf_nodes = %s' % (node_id, p.leaf_nodes)
                pass
            p.leaf_nodes.append(left)
            p.leaf_nodes.append(right)
            # copying relevant bits
            p.node_info[node_id] = tree_pg.node_info[node_id]
            p.logprior[node_id] = tree_pg.logprior[node_id]
            for node_id_child in [left, right]:
                p.do_not_split[node_id_child] = False   # can look up where node_id_child occurred in nodes_processed_itr
                p.loglik[node_id_child] = tree_pg.loglik[node_id_child]
                p.logprior[node_id_child] = tree_pg.logprior[node_id_child]
                p.train_ids[node_id_child] = tree_pg.train_ids[node_id_child]
                p.sum_y[node_id_child] = tree_pg.sum_y[node_id_child]
                p.sum_y2[node_id_child] = tree_pg.sum_y2[node_id_child]
                p.param_n[node_id_child] = tree_pg.param_n[node_id_child]
                p.n_points[node_id_child] = tree_pg.n_points[node_id_child]
        if settings.verbose >= 2:
            print 'p.leaf_nodes = %s' % p.leaf_nodes
            print 'p.non_leaf_nodes = %s' % p.non_leaf_nodes
            print 'p.node_info.keys() = %s' % sorted(p.node_info.keys())
        try:
            p.grow_nodes = tree_pg.grow_nodes_itr[itr+1]
            p.log_sis_ratio_d = tree_pg.log_sis_ratio_d
            p.depth = tree_pg.depth
        except IndexError:
            p.do_not_grow = True
    except IndexError:
        p.do_not_grow = True


def run_smc(particles, data, settings, param, log_weights, cache, tree_pg=None):
    if settings.verbose >= 2:
        print 'Conditioned tree:'
        tree_pg.print_tree()
    itr = 0
    while True:
        if settings.verbose >= 2:
            print '\n'
            print '*'*80
            print 'Current iteration = %3d' % itr
            print '*'*80
        if itr != 0:
            # no resampling required when itr == 0 since weights haven't been updated yet
            if settings.verbose >= 1:
                print 'iteration = %3d, log p(y|x) = %.2f, ess/n_particles = %f'  % (itr, log_pd, ess)
            (particles, log_weights) = resample(particles, log_weights, settings, log_pd, \
                                                    ess, weights_norm, tree_pg)
        for pid, p in enumerate(particles):
            if settings.verbose >= 2:
                print 'Current particle = %3d' % pid
                print 'grow_nodes = %s' % p.grow_nodes
                print 'leaf_nodes = %s, non_leaf_nodes = %s' % (p.leaf_nodes, p.non_leaf_nodes)
            if p.grow_nodes:
                p.grow_nodes_itr.append(p.grow_nodes[:])
            if tree_pg and (pid == 0):
                if settings.verbose >= 2 and itr == 0:
                    for s in ['leaf_nodes', 'non_leaf_nodes', 'grow_nodes_itr', 'ancestry', 'nodes_processed_itr']:
                        print 'p.%s = %s' % (s, getattr(p, s))
                grow_next_pg(p, tree_pg, itr, settings)
            else:
                p.grow_next(data, param, settings, cache)
            p.update_depth()
            if settings.verbose >= 2:
                print 'nodes_processed_itr for particle = %s' % p.nodes_processed_itr
                print 'grow_nodes (after running grow_next) (NOT updated for conditioned tree_pg) = %s' % p.grow_nodes
                print 'leaf_nodes = %s, non_leaf_nodes = %s' % (p.leaf_nodes, p.non_leaf_nodes)
                print 'nodes_processed_itr for particle (after running update_particle weights) = %s' % p.nodes_processed_itr
                print 'checking nodes_processed_itr'
        (log_pd, ess, log_weights, weights_norm) = \
                    update_particle_weights(particles, log_weights, settings)     # in place update of log_weights
        if settings.verbose >= 2:
            print 'log_weights = %s' % log_weights
        if check_do_not_grow(particles):
            if settings.verbose >= 1:
                print 'None of the particles can be grown any further; breaking out'
            break
        itr += 1
    if (settings.debug == 1) and tree_pg:
        for pid, p in enumerate(particles):
            if settings.verbose >=2 :
                print 'checking pid = %s' % pid
            p.check_nodes_processed_itr(settings)
        if settings.verbose >= 2:  
            print 'check if tree_pg did the right thing:'
            print 'nodes_processed_itr (orig, new):\n%s\n%s' % (tree_pg.nodes_processed_itr, particles[0].nodes_processed_itr)
            print 'leaf_nodes (orig, new):\n%s\n%s' % (tree_pg.leaf_nodes, particles[0].leaf_nodes)
            print 'non_leaf_nodes (orig, new):\n%s\n%s' % (tree_pg.non_leaf_nodes, particles[0].non_leaf_nodes)
            print 'grow_nodes_itr (orig, new):\n%s\n%s' % (tree_pg.grow_nodes_itr, particles[0].grow_nodes_itr)
        assert particles[0].leaf_nodes == tree_pg.leaf_nodes
        assert particles[0].non_leaf_nodes == tree_pg.non_leaf_nodes
        assert particles[0].grow_nodes_itr == tree_pg.grow_nodes_itr
    return (particles, ess, log_weights, log_pd)


def init_run_smc(data, settings, param, cache, cache_tmp, tree_pg=None):
    particles, log_weights = init_particles(data, settings, param, cache_tmp)
    (particles, ess, log_weights, log_pd) = \
           run_smc(particles, data, settings, param, log_weights, cache, tree_pg)
    return (particles, log_pd, log_weights)


def check_do_not_grow(particles):
    """ Test if all particles have grown fully """
    do_not_grow = True
    for p in particles:
        do_not_grow = do_not_grow and p.do_not_grow
    return do_not_grow
