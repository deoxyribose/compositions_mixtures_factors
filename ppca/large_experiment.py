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


def set_uninformative_priors(K,prior_std):
    scaleloc = torch.zeros(1)
    scalescale = prior_std*torch.ones(1)
    cov_factor_loc = torch.zeros(K,D)
    cov_factor_scale = prior_std*torch.ones(K,D)
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_random_variational_parameter_init(K,prior_std):
    scaleloc = torch.randn(1)
    scalescale = prior_std*torch.abs(torch.randn(1))
    cov_factor_loc = torch.randn(K,D)
    cov_factor_scale = prior_std*torch.abs(torch.randn(K,D))
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_incremental_priors(K,prior_std):
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
    cov_scale_init = prior_std*torch.ones(K,D)
    cov_scale_init[:K-1,:] = prev_posterior_scale
    return K, torch.tensor(param_history['scale_loc']),torch.tensor(param_history['scale_scale']),cov_loc_init,cov_scale_init

def set_incremental_variational_parameter_init(K,prior_std):
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
    cov_scale_init = prior_std*torch.abs(torch.randn(K,D))
    cov_scale_init[:K-1,:] = prev_posterior_scale
    return K, torch.tensor(param_history['scale_loc']),torch.tensor(param_history['scale_scale']),cov_loc_init,cov_scale_init


def get_h_and_v_params(K, experimental_condition = 0, prior_std = 1):
    if experimental_condition == 0:
        return set_uninformative_priors(K, prior_std), set_random_variational_parameter_init(K, prior_std)
    if experimental_condition == 1:
        # since incremental variational parameters are the same as the incremental prior parameters
        return set_uninformative_priors(K, prior_std), set_incremental_variational_parameter_init(K, prior_std)
    if experimental_condition == 2:
        return set_incremental_priors(K, prior_std), set_random_variational_parameter_init(K, prior_std)
    if experimental_condition == 3:
        return set_incremental_priors(K, prior_std), set_incremental_variational_parameter_init(K, prior_std)

def dgp(X): # data generating process
    N, D = X.shape
    hyperparameters,_ = get_h_and_v_params(trueK)
    K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc, scalescale))
    # ppca has sigma*I covariance
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D):
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', N):
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(torch.zeros(D), cov_factor=cov_factor, cov_diag=cov_diag))
    return X

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments for incremental inference in ppca')
    parser.add_argument('dataseed', type=int, help='Random seed for generating data')
    parser.add_argument('initseed', type=int, help='Random seed for variational parameter initialization')
    args = parser.parse_args()
    # set seed so that the same initalization and MC samples are used throughout the experiment (except the informative priors)
    pyro.set_rng_seed(args.dataseed)

    ####################
    # define parameters shared between conditions

    n_experimental_conditions = 4
    max_n_iter = 10000
    dgp_prior_std = 1
    proportion_of_data_for_testing = 0.2
    n_posterior_samples = 1600

    # optimization parameters
    learning_rate = 0.05
    momentum1 = 0.9
    momentum2 = 0.999
    decay = 1.
    batch_size = 10
    n_mc_samples = 16
    window = 10 # compute lppd every window iterations
    convergence_window = 10 # estimate slope of convergence_window lppds
    slope_significance = 0.5 # p_value of slope has to be smaller than this for training to continue

    for totalN in [1000,10000]:
        for D in [50,500]:
            for model_prior_std in [1,3]:
                ####################
                # generate data
                trueK = 4#D//3
                K = trueK
                #Kmax = 8#D//2
                Kmax = 5#D//2
                prior_std = 1
                trace = pyro.poutine.trace(dgp).get_trace(torch.zeros(totalN,D))
                logp = trace.log_prob_sum()
                true_variables = [trace.nodes[name]["value"] for name in trace.stochastic_nodes]
                _,true_scale,_,true_cov_factor,_,all_data = true_variables
                test_idxs = np.random.choice(totalN,size=int(totalN*proportion_of_data_for_testing),replace=False)
                mask = np.ones(all_data.shape[0],dtype=bool)
                mask[test_idxs] = False
                data = all_data[mask]
                test_data = all_data[~mask]
                N = data.shape[0]
                #####################
                # train models
                prior_std = model_prior_std
                for experimental_condition in range(n_experimental_conditions):
                    for K in range(1,Kmax+1):
                        pyro.set_rng_seed(args.initseed)
                        # all K=1 models, seeds and data are identical, so just load it
                        if experimental_condition > 0 and K == 1:
                            K1model = "{}_ppcas_{}_dataseed_{}_initseed_{}_N_{}_D_{}_priorstd_{}.p".format(1,0, args.dataseed, args.initseed,N,D,model_prior_std)
                            _,_,_,param_history,_ = pickle.load(open(K1model, 'rb'))
                            continue
                        elif experimental_condition == 0:
                            param_history = None
                        filename = "{}_ppcas_{}_dataseed_{}_initseed_{}_N_{}_D_{}_priorstd_{}.p".format(K,str(experimental_condition), args.dataseed, args.initseed,N,D,model_prior_std)
                        # if experiment gets interrepted, continue from loaded results
                        if os.path.exists(filename):
                            _,_,_,param_history,_ = pickle.load(open(filename, 'rb'))
                            data = trace.nodes['obs']['value']
                            print("Model has been run before, loading data and continuing.")
                            continue
                        start = time.time()
                        print('\nTraining model with {} ppcas with prior_std {} in experimental condition {} on data with {} observations in {} dimensions '.format(K, prior_std, experimental_condition, N, D))
                        pyro.clear_param_store()
                        init = get_h_and_v_params(K, experimental_condition, prior_std)
                        inference_results = inference(incrementalPpca, incrementalPpcaGuide, data, test_data, init, max_n_iter, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                        svi, losses, lppds, param_history, init, gradient_norms = inference_results
                        end = time.time()
                        print('\nTraining took {} seconds'.format(round(end - start))) 
                ########################
                # save models
                        pickle.dump((trace, losses, lppds, param_history, round(end - start)),open(filename, "wb" ))