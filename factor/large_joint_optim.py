import argparse
import numpy as np
import torch
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from pyro import distributions as dst
from collections import defaultdict
import matplotlib.pylab as plt
import time
import pickle
import os
import pandas as pd
import xarray as xr
import scipy.stats as sps
import argparse
from sklearn.decomposition import PCA
import gc
import sys
sys.path.append("..")
from tracepredictive import *
from inference import *
from models_and_guides import *

def set_uninformative_priors(K,D,prior_std):
    scaleloc = torch.zeros(D)
    scalescale = prior_std*torch.ones(D)
    cov_factor_loc = torch.zeros(K,D)
    cov_factor_scale = prior_std*torch.ones(K,D)
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_random_variational_parameter_init(K,D,prior_std):
    scaleloc = torch.randn(D)
    scalescale = torch.abs(torch.randn(D))
    #scalescale = prior_std*torch.abs(torch.randn(1))
    cov_factor_loc = torch.randn(K,D)
    cov_factor_scale = torch.abs(torch.randn(K,D))
    #cov_factor_scale = prior_std*torch.abs(torch.randn(K,D))
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_PCA_variational_parameter_init(K,D,prior_std,data):
    scaleloc = torch.randn(D)
    scalescale = torch.abs(torch.randn(D))
    #scalescale = prior_std*torch.abs(torch.randn(1))
    tmp = PCA(n_components=K)
    tmp.fit(data)
    cov_factor_loc = torch.tensor(tmp.components_,dtype = torch.float32)
    cov_factor_scale = torch.abs(torch.randn(K,D))
    #cov_factor_scale = prior_std*torch.abs(torch.randn(K,D))
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_incremental_priors(K,D,prior_std):
    print('Setting priors to posterior learnt by previous model.')
    if K == 2:
        prev_posterior_loc = torch.tensor(param_history['cov_factor_loc_{}'.format(K-1)])
        prev_posterior_scale = torch.tensor(param_history['cov_factor_scale_{}'.format(K-1)])
    else:
        cov_factor_loc = torch.tensor(param_history['cov_factor_loc_{}'.format(K-1)])
        cov_factor_scale = torch.tensor(param_history['cov_factor_scale_{}'.format(K-1)])
        cov_factor_new_loc = torch.tensor(param_history['cov_factor_new_loc_{}'.format(K-1)])
        cov_factor_new_scale = torch.tensor(param_history['cov_factor_new_scale_{}'.format(K-1)])

        prev_posterior_loc = torch.cat([cov_factor_loc,torch.unsqueeze(cov_factor_new_loc,dim=0)])
        prev_posterior_scale = torch.cat([cov_factor_scale,torch.unsqueeze(cov_factor_new_scale,dim=0)])
    cov_loc_init = torch.zeros(K,D)
    cov_loc_init[:K-1,:] = prev_posterior_loc
    cov_scale_init = torch.ones(K,D)
    #cov_scale_init = prior_std*torch.ones(K,D)
    cov_scale_init[:K-1,:] = prev_posterior_scale
    return K, torch.tensor(param_history['scale_loc']),torch.tensor(param_history['scale_scale']),cov_loc_init,cov_scale_init

def set_incremental_variational_parameter_init(K,D,prior_std):
    print('Setting variational parameters to those learnt by previous model.')
    if K == 2:
        prev_posterior_loc = torch.tensor(param_history['cov_factor_loc_{}'.format(K-1)])
        prev_posterior_scale = torch.tensor(param_history['cov_factor_scale_{}'.format(K-1)])
    else:
        cov_factor_loc = torch.tensor(param_history['cov_factor_loc_{}'.format(K-1)])
        cov_factor_scale = torch.tensor(param_history['cov_factor_scale_{}'.format(K-1)])
        cov_factor_new_loc = torch.tensor(param_history['cov_factor_new_loc_{}'.format(K-1)])
        cov_factor_new_scale = torch.tensor(param_history['cov_factor_new_scale_{}'.format(K-1)])
        prev_posterior_loc = torch.cat([cov_factor_loc,torch.unsqueeze(cov_factor_new_loc,dim=0)])
        prev_posterior_scale = torch.cat([cov_factor_scale,torch.unsqueeze(cov_factor_new_scale,dim=0)])
    cov_loc_init = torch.randn(K,D)
    cov_loc_init[:K-1,:] = prev_posterior_loc
    cov_scale_init = torch.abs(torch.randn(K,D))
    #cov_scale_init = prior_std*torch.abs(torch.randn(K,D))
    cov_scale_init[:K-1,:] = prev_posterior_scale
    return K, torch.tensor(param_history['scale_loc']),torch.tensor(param_history['scale_scale']),cov_loc_init,cov_scale_init

def get_h_and_v_params(K,D,experimental_condition = 0, prior_std = 1, data = None):
    assert D > 0
    assert K > 0
    if experimental_condition == 0:
        #return set_uninformative_priors(K, D, prior_std), set_random_variational_parameter_init(K, D, prior_std)
        return set_random_variational_parameter_init(K, D, prior_std), set_PCA_variational_parameter_init(K,D,prior_std,data)
    if experimental_condition == 1:
        # since incremental variational parameters are the same as the incremental prior parameters
    #    return set_uninformative_priors(K, D, prior_std), set_incremental_variational_parameter_init(K, D, prior_std)
        return set_incremental_priors(K, D, prior_std), set_incremental_variational_parameter_init(K, D, prior_std)
    else:
        return set_random_variational_parameter_init(K, D, prior_std), set_random_variational_parameter_init(K, D, prior_std)
    #if experimental_condition == 2:
    #    return set_incremental_priors(K, D, prior_std), set_random_variational_parameter_init(K, D, prior_std)
    #if experimental_condition == 3:
    #    return set_incremental_priors(K, D, prior_std), set_incremental_variational_parameter_init(K, D, prior_std)
    #if experimental_condition == 4:
    #    return set_uninformative_priors(K, D, prior_std), set_PCA_variational_parameter_init(K,D,prior_std,data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for incremental inference in ppca')
    parser.add_argument('dataseed', type=int, help='Random seed for generating data')
    parser.add_argument('initseed', type=int, help='Random seed for variational parameter initialization')
    args = parser.parse_args()
    # set seed so that the same initalization and MC samples are used throughout the experiment (except the informative priors)
    pyro.set_rng_seed(args.dataseed)

    ####################
    # define parameters shared between conditions

    n_experimental_conditions = 2
    max_n_iter = 800
    dgp_prior_std = 1
    proportion_of_data_for_testing = 0.2
    n_posterior_samples = 1600

    # optimization parameters
    n_multistart = 10
    learning_rate = 0.05
    momentum1 = 0.9
    momentum2 = 0.999
    decay = 1.
    batch_size = 10
    n_mc_samples = 10
    window = 10 # compute lppd every window iterations
    convergence_window = 10 # estimate slope of convergence_window lppds
    slope_significance = 1. # p_value of slope has to be smaller than this for training to continue

    for totalN in [1000,10000]:
        for D in [10,500]:#[20,30]: #
            trueK = 7#D//3
            K = trueK
            trueinit = get_h_and_v_params(K,D,experimental_condition = None, prior_std = 1)
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
            Kmax = 10#D//2
            prior_std = 1
            #####################
            # train models
            for experimental_condition in range(n_experimental_conditions):
                for K in range(1,Kmax+1):
                    pyro.set_rng_seed(args.initseed)
                    # all K=1 models, seeds and data are identical, so just load it
                    if experimental_condition > 0 and experimental_condition < 4 and K == 1:
                        K1model = "{}_factors_{}_dataseed_{}_initseed_{}_N_{}_D_{}_priorstd_{}.p".format(1,0, args.dataseed, args.initseed,N,D,prior_std)
                        _,_,_,param_history,_,_ = pickle.load(open(K1model, 'rb'))
                        continue
                    elif experimental_condition == 0:
                        param_history = None
                    filename = "{}_factors_{}_dataseed_{}_initseed_{}_N_{}_D_{}_priorstd_{}.p".format(K,str(experimental_condition), args.dataseed, args.initseed,N,D,prior_std)
                    # if experiment gets interrepted, continue from loaded results
                    if os.path.exists(filename):
                        _,_,_,param_history,_,_ = pickle.load(open(filename, 'rb'))
                        data = trace.nodes['obs']['value']
                        print("Model has been run before, loading data and continuing.")
                        continue
                    start = time.time()
                    print('\nTraining model with {} factors with prior_std {} in experimental condition {} on data with {} observations in {} dimensions '.format(K, prior_std, experimental_condition, N, D))
                    
                    best_loss_after_init = np.inf
                    inits = []
                    for restart in range(n_multistart):
                        print('Multistart {}/{}'.format(restart+1,n_multistart))
                        pyro.clear_param_store()
                        # initialize
                        init = get_h_and_v_params(K, D, experimental_condition, prior_std, data)
                        # run 300 iterations
                        inference_results = inference(zeroMeanFactor, zeroMeanFactorGuide, data, test_data, init, 300, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                        _, _, lppds, _, _,_ = inference_results
                        loss_after_init = sum(lppds[-3:])/3
                        inits.append(lppds)
                        if loss_after_init < best_loss_after_init:
                            best_loss_after_init = loss_after_init
                            best_init = init
                    init = best_init
                    inference_results = inference(zeroMeanFactor, zeroMeanFactorGuide, data, test_data, init, max_n_iter, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                    svi, losses, lppds, param_history, init, gradient_norms = inference_results
                    end = time.time()
                    print('\nTraining took {} seconds'.format(round(end - start))) 
            ########################
            # save models
                    pickle.dump((trace, losses, lppds, param_history, inits, round(end - start)),open(filename, "wb" ))