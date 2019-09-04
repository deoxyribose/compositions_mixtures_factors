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
from scipy.stats import norm
import argparse
from sklearn.decomposition import PCA
import gc

N = 10000
D = 10

def get_h_and_v_params(data = None, K = 1, hyperparameter_std = 1, experimental_condition = 0, param_history = None):
    def get_hyperparameters():
        if not experimental_condition or param_history is None:
            locloc = torch.zeros(D)
            locscale = hyperparameter_std*torch.ones(D)
            scaleloc = torch.zeros(1)
            scalescale = hyperparameter_std*torch.ones(1)
            #if data is not None:
            #    cov_factor_loc = torch.tensor(PCA(n_components=K).fit(data).components_, dtype=torch.float32)
            #else:
            #    cov_factor_loc = hyperparameter_std*torch.randn(K,D)
            cov_factor_loc = torch.zeros(K,D)
            cov_factor_scale = hyperparameter_std*torch.ones(K,D)
            return K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale
        else:
            print('Setting priors to posterior learnt by previous model.')
            if K == 2:
                prev_posterior_loc = param_history['cov_factor_loc_{}'.format(K-1)][-1].detach()
                prev_posterior_scale = param_history['cov_factor_scale_{}'.format(K-1)][-1].detach()
            else:
                prev_posterior_loc = torch.cat([param_history['cov_factor_loc_{}'.format(K-1)][-1].detach(),torch.unsqueeze(param_history['cov_factor_new_loc_{}'.format(K-1)][-1].detach(),dim=0)])
                prev_posterior_scale = torch.cat([param_history['cov_factor_scale_{}'.format(K-1)][-1].detach(),torch.unsqueeze(param_history['cov_factor_new_scale_{}'.format(K-1)][-1].detach(),dim=0)])
            cov_loc_init = torch.zeros(K,D)
            cov_loc_init[:K-1,:] = prev_posterior_loc
            cov_scale_init = hyperparameter_std*torch.ones(K,D)
            cov_scale_init[:K-1,:] = prev_posterior_scale
            return K,param_history['loc_loc'][-1].detach(),param_history['loc_scale'][-1].detach(),param_history['scale_loc'][-1].detach(),param_history['scale_scale'][-1].detach(),cov_loc_init,cov_scale_init

    def get_variational_parameter_init():
        if not experimental_condition or param_history is None:
            locloc = torch.randn(D)
            locscale = hyperparameter_std*torch.abs(torch.randn(D))
            scaleloc = torch.randn(1)
            scalescale = hyperparameter_std*torch.abs(torch.randn(1))
            #if data is not None:
            #    cov_factor_loc = torch.tensor(PCA(n_components=K).fit(data).components_, dtype=torch.float32)
            #else:
            #    cov_factor_loc = hyperparameter_std*torch.randn(K,D)
            cov_factor_loc = torch.randn(K,D)
            cov_factor_scale = hyperparameter_std*torch.abs(torch.randn(K,D))
            return K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale
        else:
            print('Initializing variational parameters to values learnt by previous model (except new factor, which is random).')
            if K == 2:
                prev_posterior_loc = param_history['cov_factor_loc_{}'.format(K-1)][-1].detach()
                prev_posterior_scale = param_history['cov_factor_scale_{}'.format(K-1)][-1].detach()
            else:
                prev_posterior_loc = torch.cat([param_history['cov_factor_loc_{}'.format(K-1)][-1].detach(),torch.unsqueeze(param_history['cov_factor_new_loc_{}'.format(K-1)][-1].detach(),dim=0)])
                prev_posterior_scale = torch.cat([param_history['cov_factor_scale_{}'.format(K-1)][-1].detach(),torch.unsqueeze(param_history['cov_factor_new_scale_{}'.format(K-1)][-1].detach(),dim=0)])
            cov_loc_init = torch.randn(K,D)
            cov_loc_init[:K-1,:] = prev_posterior_loc
            cov_scale_init = hyperparameter_std*torch.abs(torch.randn(K,D))
            cov_scale_init[:K-1,:] = prev_posterior_scale
            return K,param_history['loc_loc'][-1].detach(),param_history['loc_scale'][-1].detach(),param_history['scale_loc'][-1].detach(),param_history['scale_scale'][-1].detach(),cov_loc_init,cov_scale_init
    return get_hyperparameters(), get_variational_parameter_init()

def dgp(X): # data generating process
    N, D = X.shape
    # True number of factors that we'll be looking for is 3
    hyperparameters,_ = get_h_and_v_params(K = 3, hyperparameter_std = 1)
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc, scalescale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dst.Normal(locloc, locscale))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', N):
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag))
    return X

def model(X, batch_size, hyperparameters):
    N, D = X.shape
    hyperparameters,_ = hyperparameters
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc, scalescale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dst.Normal(locloc, locscale))
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
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X

def guide(X, batch_size, hyperparameters):
    N, D = X.shape
    _,init = hyperparameters
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc_init, cov_factor_scale_init = init
    cov_diag_loc = pyro.param('scale_loc', scaleloc)
    cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
    cov_diag = pyro.sample('scale', dst.LogNormal(cov_diag_loc, cov_diag_scale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D, dim=-1):
        loc_loc = pyro.param('loc_loc', locloc)
        loc_scale = pyro.param('loc_scale', locscale, constraint=constraints.positive)
        # sample variables
        loc = pyro.sample('loc', dst.Normal(loc_loc,loc_scale))
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
    return loc, cov_factor, cov_diag


def inference(model, guide, data, K, experimental_condition = 0, param_history = None, hyperparameter_std = 1, track_params = True, n_iter = 20000):

    def per_param_callable(module_name, param_name):
        if 'new' in param_name or not experimental_condition or K==1:
            #sgd
            #return {"lr": 1e-7, "momentum": 0.9}
            #adam
            return {"lr": 0.1, "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html
        else:
            # this is the learning rate applied to all parameters that are transferred
            # if the rate is set to 0, the parameters are transferred, and locked, so they don't change
            #sgd
            #return {"lr": 0, "momentum": 0.9}
            #adam
            return {"lr": 0, "betas": (0.90, 0.999)} 

    conditioned_model = pyro.condition(model, data = {'obs': data})
    #optim = torch.optim.SGD
    optim = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': 0.9999})
    elbo = Trace_ELBO()
    svi = SVI(model, guide, scheduler, loss=elbo, num_samples=10)
    svi2 = SVI(model, guide, scheduler, loss=elbo, num_samples=1000)

    # initialize variational parameters
    def initialize(seed):
        #pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        # Initialize new parameters
        init = get_h_and_v_params(data = None, K = K, hyperparameter_std = hyperparameter_std, experimental_condition = experimental_condition, param_history = param_history)
        # calling the guide with a clear param store initializes the params
        batch_size = N
        guide(data,batch_size,init)
        # use svi2 to have better loss estimates
        loss = svi2.loss(model, guide, data, batch_size, init)
        return loss, init
    # Choose the best among 10 random initializations.
    inits = [(initialize(seed), seed) for seed in range(1)]
    #losses = [loss for ((loss, init), seed) in inits]
    #plt.boxplot(losses)
    (loss, init), seed = min(inits)
    initialize(seed)

    # we don't need a precise ELBO after initalizing, so...
    batch_size = N//1000
    
    # Register hooks to monitor gradient norms.
    losses = []
    lrs = []
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
    window = 300
    while i < n_iter:
        loss = svi.step(data, batch_size, init)
        if i % window or i <= window:
            print('.', end='')
            scheduler.step()
            state = scheduler.get_state()['loc_loc']
            lr = state['base_lrs'][0]*state['gamma']**state['last_epoch']
        else:
            print('\nSetting lr to {} after {} iterations'.format(lr, i), end='')
            print('\n', end='')
            #raw_batch_size *= 1.05
            batch_size = min(1000,int(raw_batch_size))
            print('\nSetting batch size to {}'.format(batch_size), end='')
        losses.append(loss)
        lrs.append(lr)
        if track_params:
            # take one svi step to populate the param store
            # warning: this builds the param_history dict from scratch in every iteration
            param_history = {k:torch.cat([param_history[k],v.unsqueeze(0).detach()],dim=0) for k,v in pyro.get_param_store().items()}
            #print('.' if i % 100 else '\n', end='')
        i += 1
    print('\nConverged in {} iterations.\n'.format(i))
    params = pyro.get_param_store()
    losses = np.array(losses)
    losses = np.pad(losses, (0,n_iter-i), 'edge')
    return losses, param_history, svi2, init, lrs, gradient_norms

def posterior_predictive(model, posterior, data, hyperparameters, n_samples = 1000):
    batch_size = N
    unconditioned_model = pyro.poutine.uncondition(model)
    trace_pred = TracePredictive(unconditioned_model, posterior, num_samples=n_samples).run(data, batch_size, hyperparameters)
    random_idx = np.random.randint(batch_size)
    predictive_dst_sample = [torch.unsqueeze(trace.nodes['obs']['value'][random_idx,:],dim=0) for trace in trace_pred.exec_traces]
    predictive_dst_sample = torch.cat(predictive_dst_sample, dim=0)
    random_idx = np.random.randint(N, size = batch_size)
    model_log_evidence = [torch.unsqueeze(trace.nodes['obs']['fn'].log_prob(data[random_idx,:]),dim=0) for trace in trace_pred.exec_traces]
    model_log_evidence = (torch.logsumexp(torch.cat(model_log_evidence),dim=0)-np.log(np.float(10))).mean()
    return predictive_dst_sample, model_log_evidence

if __name__ == '__main__':
    # set seed so that the same initalization and MC samples are used throughout the experiment (except the informative priors)
    seed = 2*42
    pyro.set_rng_seed(seed)
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #cuda0 = torch.device('cuda:0')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # run dgp and get a trace of its execution to record the true parameters
    trace = pyro.poutine.trace(dgp).get_trace(torch.zeros(N,D))
    logp = trace.log_prob_sum()
    true_variables = [trace.nodes[name]["value"] for name in trace.stochastic_nodes]
    trace.stochastic_nodes
    _,true_loc,true_scale,_,true_cov_factor,_,data = true_variables
    ####################
    # start with a 1 factor model, with randomly initialized parameters
    Kmax = 10
    n_experimental_conditions = 2
    n_iter = 10000
    n_ppc_samples = 1000
    telemetry = xr.DataArray(np.zeros((Kmax,n_experimental_conditions,n_iter)))
    results = xr.DataArray(np.zeros((Kmax,n_experimental_conditions,n_ppc_samples,D)))
    model_log_evidences = xr.DataArray(np.zeros((Kmax, n_experimental_conditions, 1)))
    ####################
    # start with a 1 factor model, with randomly initialized parameters
    for experimental_condition in range(n_experimental_conditions):
        for K in range(1,Kmax+1):
            if os.path.exists("ppca_experiment_transfer_{}_ppcas_{}_seed_{}.p".format(K,str(experimental_condition), seed)):
                data, _, param_history, _, _, _, _ = pickle.load(open("ppca_experiment_transfer_{}_ppcas_{}_seed_{}.p".format(K,str(experimental_condition), seed), 'rb'))
                print("Model has been run before, loading data and continuing.")
                continue
            start = time.time()
            print('\nTraining model with {} ppcas'.format(K))
            if K == 1:
                param_history = None
            learning_curve, param_history, posterior, init, lrs, gradient_norms = inference(model, guide, data, K, experimental_condition, param_history = param_history, hyperparameter_std = 3, n_iter = n_iter)
            predictive_posterior_sample, model_log_evidence = posterior_predictive(model, posterior, data, init)
            # save experimental data
            #telemetry[[K-1,experimental_condition]] = learning_curve
            #results[[K-1,experimental_condition]] = predictive_posterior_sample.detach().numpy()
            #model_log_evidences[[K-1,experimental_condition]] = model_log_evidence.detach().numpy()
            end = time.time()
            print('\nTraining took {} seconds'.format(round(end - start))) 
            pickle.dump((data, learning_curve, param_history, predictive_posterior_sample, model_log_evidence, lrs, gradient_norms),open("ppca_experiment_transfer_{}_ppcas_{}_seed_{}.p".format(K,str(experimental_condition), seed), "wb" ))
            gc.collect()