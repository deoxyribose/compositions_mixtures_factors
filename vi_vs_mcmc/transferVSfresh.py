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

N = 10000
D = 10

def dgp(X, device): # data generating process
    N, D = X.shape
    K = 7 # True number of factors that we'll be looking for
    #K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = hyperparameters
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = (K,torch.randn(D),torch.abs(torch.randn(D)),torch.randn(D),torch.abs(torch.randn(D)),torch.randn(K,D),torch.abs(torch.randn(K,D))*2)
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dst.Normal(locloc.to(device), locscale.to(device)))
        cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc.to(device), scalescale.to(device)))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc.to(device),cov_factor_scale.to(device)))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', N):
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag))
    return X

def model(X, hyperparameters):
    N, D = X.shape
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale, device = hyperparameters
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dst.Normal(locloc.to(device), locscale.to(device)))
        cov_diag = pyro.sample('scale', dst.LogNormal(scaleloc.to(device), scalescale.to(device)))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc.to(device),cov_factor_scale.to(device)))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', N):
        X = pyro.sample('obs', dst.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag))
    return X

def guide(X, hyperparameters):
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale, device = hyperparameters
    with pyro.plate('D', D, dim=-1):
        loc_loc = pyro.param('loc_loc', locloc.to(device))
        loc_scale = pyro.param('loc_scale', locscale.to(device), constraint=constraints.positive)
        cov_diag_loc = pyro.param('scale_loc', scaleloc.to(device))
        cov_diag_scale = pyro.param('scale_scale', scalescale.to(device), constraint=constraints.positive)
        # sample variables
        loc = pyro.sample('loc', dst.Normal(loc_loc,loc_scale))
        with pyro.plate('K', K, dim=-2):
            cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc)
            cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale, constraint=constraints.positive)
            cov_factor = pyro.sample('cov_factor', dst.Normal(cov_factor_loc, cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
        cov_diag = pyro.sample('scale', dst.LogNormal(cov_diag_loc, cov_diag_scale))
    return loc, cov_factor, cov_diag


def inference(model, guide, data, hyperparameters, track_params = True, n_iter = 10000):
    def per_param_callable(module_name, param_name):
        return {"lr": 0.01, "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html

    pyro.clear_param_store()

    conditioned_model = pyro.condition(model, data = {'obs': data})
    optim = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': 0.99 })
    elbo = Trace_ELBO()
    svi = SVI(conditioned_model, guide, scheduler, loss=elbo, num_samples=50)

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
    for i in range(n_iter):
        loss = svi.step(data, hyperparameters)
        if i % 100:
            print('.', end='')
        else:
            scheduler.step()
            state = scheduler.get_state()['loc_loc']
            print('\n {}/{}'.format(i,n_iter))
            print('\nSetting lr to {}'.format(state['base_lrs'][0]*state['gamma']**state['last_epoch']), end='')
            print('\n', end='')
        #loss = svi.step(data, hyperparameters)
        losses.append(loss)
        if track_params:
            # take one svi step to populate the param store
            # warning: this builds the param_history dict from scratch in every iteration
            param_history = {k:torch.cat([param_history[k],v.unsqueeze(0)],dim=0) for k,v in pyro.get_param_store().items()}
            #print('.' if i % 100 else '\n', end='')
    params = pyro.get_param_store()
    return losses, param_history, svi

def posterior_predictive(model, posterior, data, hyperparameters, n_samples = 1000):
    trace_pred = TracePredictive(model, posterior, num_samples=n_samples).run(data[:,:500], hyperparameters)
    random_idx = np.random.randint(n_samples)
    predictive_dst_sample = [torch.unsqueeze(trace.nodes['obs']['value'][random_idx,:],dim=0) for trace in trace_pred.exec_traces]
    predictive_dst_sample = torch.cat(predictive_dst_sample, dim=0)
    return predictive_dst_sample

if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    cuda0 = torch.device('cuda:0')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # run dgp and get a trace of its execution to record the true parameters
    trace = pyro.poutine.trace(dgp).get_trace(torch.zeros(N,D))
    logp = trace.log_prob_sum()
    true_variables = [trace.nodes[name]["value"] for name in trace.stochastic_nodes]
    trace.stochastic_nodes

    _,true_loc,true_scale,_,true_cov_factor,_,data = true_variables

    ####################
    # start with a 1 factor model, with randomly initialized parameters
    learning_curves = []
    param_histories = []
    predictive_posterior_samples = []
    for K in range(1,8):
        print('\nTraining model with {} factors'.format(K))
        start = time.time()
        initial_hyperparameters = (K,torch.randn(D),torch.abs(torch.randn(D)),torch.randn(D),torch.abs(torch.randn(D)),torch.randn(K,D),torch.abs(torch.randn(K,D))*2)
        learning_curve, param_history, posterior = inference(model, guide, data, initial_hyperparameters, n_iter = 10000)
        predictive_posterior_sample = posterior_predictive(model, posterior, data, initial_hyperparameters)
        learning_curves.append(learning_curve)
        param_histories.append(param_history)
        predictive_posterior_samples.append(predictive_posterior_sample)
        end = time.time()
        print('\nTraining took {} seconds'.format(round(end - start)))

    pickle.dump((data, learning_curves, param_histories, predictive_posterior_samples),open( "factor_experiment_no_transfer.p", "wb" ))