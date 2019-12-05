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
from tracepredictive import *
import time
import pickle
import os
import pandas as pd
import xarray as xr
import scipy.stats as sps
import argparse
from sklearn.decomposition import PCA
import gc

def set_uninformative_priors():
    scaleloc = torch.zeros(1)
    scalescale = prior_std*torch.ones(1)
    cov_factor_loc = torch.zeros(K,D)
    cov_factor_scale = prior_std*torch.ones(K,D)
    return scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_incremental_priors():
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
    return torch.tensor(param_history['scale_loc']),torch.tensor(param_history['scale_scale']),cov_loc_init,cov_scale_init

def set_incremental_variational_parameter_init():
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
    return torch.tensor(param_history['scale_loc']),torch.tensor(param_history['scale_scale']),cov_loc_init,cov_scale_init

def set_random_variational_parameter_init():
    scaleloc = torch.randn(1)
    scalescale = torch.abs(torch.randn(1))
    cov_factor_loc = torch.randn(K,D)
    cov_factor_scale = torch.abs(torch.randn(K,D))
    return scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def get_h_and_v_params(experimental_condition = 0):
    if experimental_condition == 0:
        return set_uninformative_priors(), set_random_variational_parameter_init()
    if experimental_condition == 1:
        # since incremental variational parameters are the same as the incremental prior parameters
        return set_uninformative_priors(), set_incremental_variational_parameter_init()
    if experimental_condition == 2:
        return set_incremental_priors(), set_random_variational_parameter_init()
    if experimental_condition == 3:
        return set_incremental_priors(), set_incremental_variational_parameter_init()

def dgp(X): # data generating process
    N, D = X.shape
    hyperparameters,_ = get_h_and_v_params()
    scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
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

def model(X, batch_size, hyperparameters):
    N, D = X.shape
    hyperparameters,_ = hyperparameters
    scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc, scalescale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D):
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1):
                cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc[:K-1,:],cov_factor_scale[:K-1,:]))
            cov_factor_new = pyro.sample('cov_factor_new', dst.Normal(cov_factor_loc[-1,:],cov_factor_scale[-1,:]))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(torch.zeros(D), cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X

def guide(X, batch_size, hyperparameters):
    N, D = X.shape
    _,init = hyperparameters
    scaleloc, scalescale, cov_factor_loc_init, cov_factor_scale_init = init
    cov_diag_loc = pyro.param('scale_loc', scaleloc)
    cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
    cov_diag = pyro.sample('scale', dst.LogNormal(cov_diag_loc, cov_diag_scale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D, dim=-1):
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1, dim=-2):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init[:K-1,:])
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor_new_loc = pyro.param('cov_factor_new_loc_{}'.format(K), cov_factor_loc_init[-1,:])
            cov_factor_new_scale = pyro.param('cov_factor_new_scale_{}'.format(K), cov_factor_scale_init[-1,:], constraint=constraints.positive)
            cov_factor_new = pyro.sample('cov_factor_new', dst.Normal(cov_factor_new_loc,cov_factor_new_scale))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init)
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    return cov_factor, cov_diag

def p_value_of_slope(loss, window):
    if len(loss) < window:
        return 0
    else:
        recent = loss[-window:]
        return sps.linregress(np.arange(window),recent)[3]

def inference(model, guide, data, K, experimental_condition = 0, param_history = None, prior_std = 1, track_params = True, n_iter = 20000, batch_size = 10, n_lppd_samples = 1600):

    def per_param_callable(module_name, param_name):
        if 'new' in param_name or not experimental_condition or K==1:
            return {"lr": initial_learning_rate, "betas": (momentum1, momentum2)}
        else:
            # this is the learning rate applied to all parameters that are transferred
            return {"lr": initial_learning_rate, "betas": (momentum1, momentum2)} 

    conditioned_model = pyro.condition(model, data = {'obs': data})
    optim = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': decay_exponent})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, scheduler, loss=elbo, num_samples=n_elbo_mc_samples)

    def initialize():
        pyro.clear_param_store()
        init = get_h_and_v_params(experimental_condition)
        # calling the guide with a clear param store initializes the params
        guide(data,batch_size,init)
        # but we also return the initialization for when we call the model and guide
        return init
    
    init = initialize()
    
    # Register hooks to monitor gradient norms.
    losses = []
    gradient_norms = defaultdict(list)
    if track_params:
        loss = svi.step(data, batch_size, init)
        param_history = dict({k:v.unsqueeze(0) for k,v in pyro.get_param_store().items()})
    # register gradient hooks for monitoring
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    # optimize
    i = 0
    raw_batch_size = batch_size
    max_n_lppd_samples = n_lppd_samples
    lppds = []
    #n_lppd_samples = n_lppd_samples//8

    #with torch.no_grad():
    #    lppd = compute_lppd(model, test_data, init, n_samples=n_lppd_samples)
    #    lppds.append(-lppd)
    #    print(lppds[-1])
    # we train if the slope is significantly different from 0
    # we prefer to infer that the slope is non-zero even if it is zero
    # to inferring that the slope is 0 when it isn't
    # so we'd rather train too much than too little
    while p_value_of_slope(lppds,convergence_window) < slope_pvalue_significance and i < n_iter:
        loss = svi.step(data, batch_size, init)
        if i % window or i <= window:
            print('.', end='')
            scheduler.step()
        else:
            with torch.no_grad():
                lppd = compute_lppd(model, test_data, init, n_samples=n_lppd_samples)
                lppds.append(-lppd)
                print(lppds[-1])

            n_lppd_samples += 8*4
            n_lppd_samples = min(n_lppd_samples, max_n_lppd_samples)

            raw_batch_size += 2
            batch_size = min(64,int(raw_batch_size))

            svi.num_samples += 1
            svi.num_samples = min(64,svi.num_samples)

            print('\nSetting number of MC samples to {}'.format(svi.num_samples), end='')
            print('\nSetting number of posterior samples {}'.format(n_lppd_samples), end='')
            print('\nSetting batch size to {}'.format(batch_size), end='')
        losses.append(loss)
        if track_params:
            # take one svi step to populate the param store
            # warning: this builds the param_history dict from scratch in every iteration
            param_history = {k:torch.cat([param_history[k],v.unsqueeze(0).detach()],dim=0) for k,v in pyro.get_param_store().items()}
        i += 1
    print('\nConverged in {} iterations.\n'.format(i))
    params = pyro.get_param_store()
    # make all pytorch tensors into np arrays, which consume less disk space
    param_history = dict(zip(param_history.keys(),map(lambda x: x.detach().numpy(), param_history.values())))
    # save just last values, since that's all we need for incremental inference, and the rest fills too much space
    last_params = dict(zip(param_history.keys(), map(lambda x: x[-1], param_history.values())))
    return losses, lppds, last_params, init, gradient_norms

def compute_lppd(model, data, hyperparameters, n_samples = 800):
    unconditioned_model = pyro.poutine.uncondition(model)
    dummy_obs = data[0:1,:]
    n_blocks = 8
    predictive_probs = torch.zeros((n_blocks, data.shape[0]))
    block_length = n_samples//n_blocks
    log_probs = torch.zeros((block_length, data.shape[0]))
    for block in range(8):
        for model_idx in range(block_length):
            guide_trace = pyro.poutine.trace(guide).get_trace(dummy_obs, 1, hyperparameters)
            blockreplay = pyro.poutine.block(fn = pyro.poutine.replay(unconditioned_model, guide_trace),expose=['loc','obs'])
            posterior_predictive = pyro.poutine.trace(blockreplay).get_trace(dummy_obs, 1, hyperparameters)
            log_probs[model_idx] = posterior_predictive.nodes['obs']['fn'].log_prob(data)
        probs = torch.logsumexp(log_probs,dim=0)-np.log(np.float(block_length))
        predictive_probs[block] = probs
    return (torch.logsumexp(predictive_probs,dim=0)-np.log(np.float(n_blocks))).mean()

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
    n_lppd_samples = 1600

    # optimization parameters
    initial_learning_rate = 0.1
    momentum1 = 0.9
    momentum2 = 0.999
    decay_exponent = 0.99995
    batch_size = 10
    n_elbo_mc_samples = 10
    window = 50 # compute lppd every window iterations
    convergence_window = 10 # estimate slope of convergence_window lppds
    slope_pvalue_significance = 0.5 # p_value of slope has to be smaller than this for training to continue

    for totalN in [1000,10000]:
        for D in [50,500]:
            for model_prior_std in [1,3]:
                ####################
                # generate data
                trueK = 4#D//3
                K = trueK
                Kmax = 8#D//2
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
                        inference_results = inference(model, guide, data, K, experimental_condition, param_history = param_history, prior_std = model_prior_std, n_iter = max_n_iter)
                        losses, lppds, param_history, init, gradient_norms = inference_results
                        end = time.time()
                        print('\nTraining took {} seconds'.format(round(end - start))) 
                ########################
                # save models
                        pickle.dump((trace, losses, lppds, param_history, round(end - start)),open(filename, "wb" ))
