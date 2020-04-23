import argparse
import numpy as np
import torch
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from pyro import distributions as dist
from collections import defaultdict
import matplotlib.pylab as plt
import time
import pickle
import os
import pandas as pd
import xarray as xr
import scipy.stats as sps
import argparse
import gc
import sys
sys.path.append("..")
from tracepredictive import *
from inference import *
from models_and_guides import *
from initializations import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for incremental inference in ppca')
    parser.add_argument('dataseed', type=int, help='Random seed for generating data')
    parser.add_argument('initseed', type=int, help='Random seed for variational parameter initialization')
    parser.add_argument('smoke_test', type=bool, nargs='?', const = True, default=False)
    args = parser.parse_args()
    # set seed so that the same initalization and MC samples are used throughout the experiment (except the informative priors)
    pyro.set_rng_seed(args.dataseed)

    ####################
    # define parameters shared between conditions

    smoke_test = args.smoke_test

    if smoke_test:
        n_conditions = 3
        max_n_iter = 20
        dgp_prior_std = 1
        proportion_of_data_for_testing = 0.1
        n_posterior_samples = 100

        # optimization parameters
        n_multistart = 2
        learning_rate = 0.05
        momentum1 = 0.9
        momentum2 = 0.999
        decay = 1.
        batch_size = 10
        n_mc_samples = 2
        window = 10 # compute lppd every window iterations
        convergence_window = 10 # estimate slope of convergence_window lppds
        slope_significance = 1. # p_value of slope has to be smaller than this for training to continue
        lowN = 50
        lowD = 5
        Kmax = 2
    else:
        n_conditions = 3
        max_n_iter = 1500
        dgp_prior_std = 1
        proportion_of_data_for_testing = 0.1
        n_posterior_samples = 1000

        # optimization parameters
        n_multistart = 6
        learning_rate = 0.05
        momentum1 = 0.9
        momentum2 = 0.999
        decay = 1.
        batch_size = 10
        n_mc_samples = 10
        window = 3 # compute lppd every window iterations
        convergence_window = 10 # estimate slope of convergence_window lppds
        slope_significance = 1. # p_value of slope has to be smaller than this for training to continue
        lowN = 5000
        lowD = 50
        Kmax = 10

    for totalN in [lowN,lowN*2]:
        for D in [lowD,lowD*10]:
            print("Running experiment with {} {}-dimensional observations".format(totalN, D))
            trueK = 7#D//3
            trueinit = get_h_and_v_params(trueK,D,condition = None, prior_std = 1)
            dgp = pyro.poutine.uncondition(zeroMeanFactor)
            trace = pyro.poutine.trace(dgp).get_trace(torch.zeros(totalN,D),totalN,trueinit)
            logp = trace.log_prob_sum()
            true_variables = dict([(name,trace.nodes[name]["value"]) for name in trace.stochastic_nodes if len(name)>1])
            all_data = true_variables['obs']
            test_idxs = np.random.choice(totalN,size=int(totalN*proportion_of_data_for_testing),replace=False)
            mask = np.ones(all_data.shape[0],dtype=bool)
            mask[test_idxs] = False 
            data = all_data[mask]
            test_data = all_data[~mask]
            N = data.shape[0]
            ####################
            # generate data
            #Kmax = 8#D//2
            prior_std = 1
            #####################
            # train models
            for condition in range(n_conditions):
                for K in range(1,Kmax+1):
                    pyro.set_rng_seed(args.initseed)
                    filename = "{}_factors_{}_dataseed_{}_initseed_{}_N_{}_D_{}_priorstd_{}.p".format(K,condition, args.dataseed, args.initseed,N,D,prior_std)

                    # if experiment gets interrepted, continue from loaded results
                    if os.path.exists(filename):
                        _,_,param_history,_,_ = pickle.load(open(filename, 'rb'))
                        data = trace.nodes['obs']['value']
                        print("Model has been run before, loading from {}.".format(filename))
                        continue
                    start = time.time()

                    print('\nTraining model with {} factors with prior_std {} in experimental condition {} on data with {} observations in {} dimensions '.format(K, prior_std, condition, N, D))
                    
                    # if we are in the incremental condition, load best restart from previous model
                    if condition == 1:
                        if K == 1:
                            continue
                        elif K ==2:
                            # load best restart from K=1, condition=0
                            c = 0
                        else:
                            # load best restart from K-1, condition=1
                            c = 1
                        prevmodel = "{}_factors_{}_dataseed_{}_initseed_{}_N_{}_D_{}_priorstd_{}.p".format(K-1,c, args.dataseed, args.initseed,N,D,prior_std)
                        param_history = get_param_history_of_best_restart(prevmodel)
                        print(prevmodel)
                    elif condition == 0:
                        param_history = None

                    best_loss_after_init = np.inf
                    inference_results_for_all_restarts = []
                    for restart in range(n_multistart):
                        restart_start = time.time()
                        print('Multistart {}/{}'.format(restart+1,n_multistart))
                        pyro.clear_param_store()
                        # initialize
                        init = get_h_and_v_params(K, D, condition, prior_std, data, param_history)
                        # train
                        inference_results = inference(zeroMeanFactor2, zeroMeanFactorGuide, data, test_data, init, max_n_iter, window, convergence_window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                        print("Restart took {} seconds".format(time.time()-restart_start))
                        inference_results_for_all_restarts.append(inference_results)
                    #    if loss_after_init < best_loss_after_init:
                    #        best_loss_after_init = loss_after_init
                    #        best_init = init
                    #init = best_init
                    #inference_results = inference(zeroMeanFactor2, zeroMeanFactorGuide, data, test_data, init, max_n_iter, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                    #svi, losses, lppds, param_history, init, gradient_norms = inference_results
                    end = time.time()
                    print('\nTraining took {} seconds'.format(round(end - start))) 
            ########################
            # save models
                    pickle.dump(inference_results_for_all_restarts,open(filename, "wb" ))
    print("All training done.")