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

N = 10000
D = 10
batch_size = N//1000

def get_hyperparameters(K = 1, hyperparameter_std = 1, experimental_condition = 0, param_history = None):
    if not experimental_condition:
        locloc = hyperparameter_std*torch.randn(D)
        locscale = torch.abs(hyperparameter_std*torch.randn(D))
        scaleloc = hyperparameter_std*torch.randn(D)
        scalescale = torch.abs(hyperparameter_std*torch.randn(D))
        cov_factor_loc = torch.tensor(PCA(n_components=K).fit(data).components_, dtype=torch.float32)
        #cov_factor_loc = hyperparameter_std*torch.randn(K,D)
        cov_factor_scale = torch.abs(hyperparameter_std*torch.randn(K,D))
        return K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale
    else:
        print('Setting priors to posterior learnt by previous model.')
        cov_loc_init = torch.randn(K,D)
        cov_loc_init[:K-1,:] = param_history['cov_factor_loc_{}'.format(K-1)][-1].detach()
        cov_scale_init = torch.abs(torch.randn(K,D))*2
        cov_scale_init[:K-1,:] = param_history['cov_factor_scale_{}'.format(K-1)][-1].detach()
        return K,param_history['loc_loc'][-1].detach(),param_history['loc_scale'][-1].detach(),param_history['scale_loc'][-1].detach(),param_history['scale_scale'][-1].detach(),cov_loc_init,cov_scale_init


def dgp(X): # data generating process
    N, D = X.shape
    # True number of factors that we'll be looking for is 7
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = get_hyperparameters(K = 7, hyperparameter_std = 3)
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dst.Normal(locloc, locscale))
        cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc, scalescale))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', N):
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag))
    return X

def model(X, hyperparameters):
    N, D = X.shape
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dst.Normal(locloc, locscale))
        cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc, scalescale))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=data.index_select(0, ind))
    return X

def guide(X, hyperparameters):
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    with pyro.plate('D', D, dim=-1):
        loc_loc = pyro.param('loc_loc', locloc)
        loc_scale = pyro.param('loc_scale', locscale, constraint=constraints.positive)
        cov_diag_loc = pyro.param('scale_loc', scaleloc)
        cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
        # sample variables
        loc = pyro.sample('loc', dst.Normal(loc_loc,loc_scale))
        with pyro.plate('K', K, dim=-2):
            cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc)
            cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale, constraint=constraints.positive)
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc, cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
        cov_diag = pyro.sample('scale', dst.LogNormal(cov_diag_loc, cov_diag_scale))
    return loc, cov_factor, cov_diag


def inference(model, guide, data, hyperparameters, track_params = True, n_iter = 20000):
    def per_param_callable(module_name, param_name):
        return {"lr": 0.1, "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html
        #return {"lr": 0.01, "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html

    pyro.clear_param_store()

    optim = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': 0.9 })
    elbo = Trace_ELBO()
    svi = SVI(model, guide, scheduler, loss=elbo, num_samples=10)
    svi2 = SVI(model, guide, scheduler, loss=elbo, num_samples=1000)

    # Register hooks to monitor gradient norms.
    losses = []
    gradient_norms = defaultdict(list)
    if track_params:
        loss = svi.step(data, hyperparameters)
        param_history = dict({k:v.unsqueeze(0) for k,v in pyro.get_param_store().items()})
    # register gradient hooks for monitoring
    for name, value in pyro.get_param_store().named_parameters():
        value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    # optimize
    i = 0
    window = 300
    gradients_are_decreasing = [True]
    while i < n_iter and any(gradients_are_decreasing):
        loss = svi.step(data, initial_hyperparameters)
        if i % window or i <= window:
            print('.', end='')
        else:
            scheduler.step()
            state = scheduler.get_state()['loc_loc']
            print('\nSetting lr to {}'.format(state['base_lrs'][0]*state['gamma']**state['last_epoch']), end='')
            print('\n', end='')
            # check if gradients are trending downwards
            gradients_are_decreasing = []
            for name, gradient_norm in gradient_norms.items():
                recent = gradient_norm[-window:]
                # estimate slope by least squares
                m_hat = np.linalg.lstsq(np.vstack([np.arange(window), np.ones(window)]).T,recent, rcond=None)[0][0]
                # estimate standard deviation of losses in window
                s_hat = np.array(recent).std(ddof=2)
                # calculate probability that slope is less than 0
                P_negative_slope = norm.cdf(0,loc=m_hat,scale=12*s_hat**2/(window**3-window))
                # if it is more than .5, loss has been decreasing
                gradients_are_decreasing.append(P_negative_slope > .5)
        losses.append(loss)
        if track_params:
            # take one svi step to populate the param store
            # warning: this builds the param_history dict from scratch in every iteration
            param_history = {k:torch.cat([param_history[k],v.unsqueeze(0)],dim=0) for k,v in pyro.get_param_store().items()}
            #print('.' if i % 100 else '\n', end='')
        i += 1
    print('\nConverged in {} iterations.\n'.format(i))
    params = pyro.get_param_store()
    losses = np.array(losses)
    losses = np.pad(losses, (0,n_iter-i), 'edge')
    return losses, param_history, svi2

def posterior_predictive(model, posterior, data, hyperparameters, n_samples = 1000):
    trace_pred = TracePredictive(model, posterior, num_samples=n_samples).run(data, hyperparameters)
    random_idx = np.random.randint(batch_size)
    predictive_dst_sample = [torch.unsqueeze(trace.nodes['obs']['value'][random_idx,:],dim=0) for trace in trace_pred.exec_traces]
    predictive_dst_sample = torch.cat(predictive_dst_sample, dim=0)
    random_idx = np.random.randint(N, size = batch_size)
    model_log_evidence = [torch.unsqueeze(trace.nodes['obs']['fn'].log_prob(data[random_idx,:]),dim=0) for trace in trace_pred.exec_traces]
    model_log_evidence = (torch.logsumexp(torch.cat(model_log_evidence),dim=0)-np.log(np.float(10))).mean()
    return predictive_dst_sample, model_log_evidence

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train a factor model on data simulated from a factor model.')
    parser.add_argument('factors', metavar='K', type=int,
                    help='Number of factors in factor model.')


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
    n_iter = 20000
    n_ppc_samples = 1000
    telemetry = xr.DataArray(np.zeros((Kmax,n_experimental_conditions,n_iter)))
    results = xr.DataArray(np.zeros((Kmax,n_experimental_conditions,n_ppc_samples,D)))
    model_log_evidences = xr.DataArray(np.zeros((Kmax, n_experimental_conditions, 1)))

    ####################
    # start with a 1 factor model, with randomly initialized parameters
    for experimental_condition in range(n_experimental_conditions):
        for K in range(1,Kmax+1):
            start = time.time()
            print('\nTraining model with {} factors'.format(K))
            tries = 0
            while tries < 4:
                try:
                    if not experimental_condition: # no transfer
                        # train every model with randomly initialized parameters
                        initial_hyperparameters = get_hyperparameters(K = K, hyperparameter_std = 3)
                    else:
                        # train every subsequent model with parameters initialized in previous ones (except the new factor parameters, which are random)
                        # this goes for both hyperparameters, and initial values of variational parameters
                        if K == 1:
                            initial_hyperparameters = get_hyperparameters(K = K, hyperparameter_std = 3)
                        else:
                            initial_hyperparameters = get_hyperparameters(K = K, hyperparameter_std = 3, experimental_condition = experimental_condition, param_history = param_history)

                    learning_curve, param_history, posterior = inference(model, guide, data, initial_hyperparameters, n_iter = n_iter)
                    break
                except RuntimeError:
                    tries += 1
                    print('Cholesky failed')

            predictive_posterior_sample, model_log_evidence = posterior_predictive(model, posterior, data, initial_hyperparameters)
            # save experimental data
            #telemetry[[K-1,experimental_condition]] = learning_curve
            #results[[K-1,experimental_condition]] = predictive_posterior_sample.detach().numpy()
            #model_log_evidences[[K-1,experimental_condition]] = model_log_evidence.detach().numpy()
            end = time.time()
            print('\nTraining took {} seconds'.format(round(end - start))) 
            pickle.dump((data, learning_curve, param_history, predictive_posterior_sample, model_log_evidence),open( "factor_experiment_transfer_{}_factors_{}.p".format(K,str(experimental_condition)), "wb" ))
    
    #pickle.dump((data, telemetry, results, model_log_evidences),open( "factor_experiment_transfer_{}_factors.p", "wb" ))
