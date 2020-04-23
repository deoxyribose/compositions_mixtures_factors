import numpy as np
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pylab as plt
from IPython.display import display, clear_output
import torch
import scipy.stats as sps
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pyro
import sys
import pyro.optim
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.predictive import _predictive, _guess_max_plate_nesting
from torch.distributions import constraints
from pyro import distributions as dst
from collections import defaultdict
from functools import wraps
import time
import sys
sys.path.append("..")
from initializations import *

def p_value_of_slope(loss, window, slope_significance):
    if len(loss) < window or any(map(torch.isinf, loss)) or slope_significance == 1:
        return 0
    else:
        recent = loss[-window:]
        return sps.linregress(np.arange(window),recent)[3]

def get_lppd(model, guide, data, init, n_samples = 1000, verbose = True):
    """
    Parallel implementation, where samples from the posterior are batched. Doesn't work for big datasets due to low_rank_multivariate_normal's log_prob having to matmul large matrices
    Slower than looping over posterior samples as in compute_lppd
    """
    # sample N latent variables from guide
    posterior_samples = _predictive(guide, {}, n_samples, parallel = True, model_args = [data[0:1,:], 1, init])
    # posterior_samples['scale'] is by default a (n_samples,1,D), needs to be (n_samples,D) or we get a Cartesian product
    posterior_samples['scale'] = torch.squeeze(posterior_samples['scale'])
    unconditioned_model = pyro.poutine.uncondition(model)
    pred = pyro.poutine.condition(unconditioned_model, posterior_samples)
    pred_trace = pyro.poutine.trace(pred).get_trace(torch.empty((n_samples,data.shape[1])), n_samples, init)
    log_probs = torch.empty((data.shape[0],n_samples))
    n_partitions = 50
    for i in range(n_partitions):
        idx0 = i*data.shape[0]//n_partitions
        idx1 = (i+1)*data.shape[0]//n_partitions
        log_probs[idx0:idx1,:] = pred_trace.nodes['obs']['fn'].log_prob(torch.unsqueeze(data[idx0:idx1,:], dim=-2))
    return (log_probs.logsumexp(dim=1)-np.log(n_samples)).mean()

def compute_lppd(model, guide, data, init, n_samples = 1000):
    unconditioned_model = pyro.poutine.uncondition(model)
    dummy_obs = data[0:1,:]
    log_probs = torch.zeros((data.shape[0],n_samples))
    for model_idx in range(n_samples):
        guide_trace = pyro.poutine.trace(guide).get_trace(dummy_obs, 1, init)
        blockreplay = pyro.poutine.block(fn = pyro.poutine.replay(unconditioned_model, guide_trace),expose=['obs'])
        posterior_predictive = pyro.poutine.trace(blockreplay).get_trace(dummy_obs, 1, init)
        log_probs[:,model_idx] = posterior_predictive.nodes['obs']['fn'].log_prob(data)
    return (torch.logsumexp(log_probs,dim=1)-np.log(np.float(n_samples))).mean()

def inference(model, guide, training_data, test_data, init, n_iter = 10000, window = 500, convergence_window = 30, batch_size = 10, n_mc_samples = 10, learning_rate = 0.1, learning_rate_decay = 0.9999, n_posterior_samples = 800, slope_significance = 0.5, track_params = False):
    pyro.clear_param_store()
    initcopy = clone_init(init)
    monitor_gradients = False
    #def per_param_callable(module_name, param_name):
    #    return {"lr": learning_rate, "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html
    optim = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': {"lr": learning_rate, "betas": (0.90, 0.999)}, 'gamma': learning_rate_decay})
    #scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': learning_rate_decay})
    if 'Mixture' in model.__repr__():
        elbo = TraceEnum_ELBO(max_plate_nesting=3)
    else:
        elbo = Trace_ELBO(max_plate_nesting=3, num_particles=n_mc_samples, vectorize_particles=True)
    svi = SVI(model, guide, scheduler, loss=elbo)

    # Register hooks to monitor gradient norms.
    losses = []
    lrs = []
    loss = svi.step(training_data, batch_size, init)
    param_history = dict({k:v.unsqueeze(0) for k,v in pyro.get_param_store().items()})
    
    if monitor_gradients:
        gradient_norms = defaultdict(list)
        # register gradient hooks for monitoring
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    # print current job
    print("Training {} with {}".format(model.__repr__(), init[0][0]))
    # optimize
    start = time.time()
    i = 0
    lppds = []
    with torch.no_grad():
#        start = time.time()
#        lppd = get_lppd2(model, guide, test_data, init, n_samples=n_posterior_samples)
#        print("Computing lppd took {}".format(time.time() - start))

        #start = time.time()
        lppd = compute_lppd(model, guide, test_data, init, n_samples=n_posterior_samples)
        #print("Computing lppd took {}".format(time.time() - start))
        lppds.append(-lppd)
        print("NLL at init is {}".format(lppds[-1]))

    while p_value_of_slope(lppds,convergence_window, slope_significance) < slope_significance and i < n_iter:
        try:
            loss = svi.step(training_data, batch_size, init)
            if i % window or i <= window:
                print('.', end='')
                scheduler.step()
            else:
                with torch.no_grad():
                    #start = time.time()
                    lppd = compute_lppd(model, guide, test_data, init, n_samples=n_posterior_samples)
                    #print("Computing lppd took {}".format(time.time() - start))
                    lppds.append(-lppd)
                    print('\n')
                    print("NLL after {}/{} iterations is {}".format(i,n_iter, lppds[-1]))
                print('\n')
                #print('\nSetting number of posterior samples to {}'.format(n_posterior_samples), end='')
                #print('\nSetting batch size to {}'.format(batch_size), end='')
            losses.append(loss)
            #lrs.append(lr)
            # take one svi step to populate the param store
            # warning: this builds the param_history dict from scratch in every iteration
            param_history = {k:torch.cat([param_history[k],v.unsqueeze(0).detach()],dim=0) for k,v in pyro.get_param_store().items()}
            i += 1
        except KeyboardInterrupt:
            print('\Interrupted by user after {} iterations.\n'.format(i))
            return svi, losses, lppds, param_history, init, gradient_norms, round(time.time() - start)
    print('\nConverged in {} iterations.\n'.format(i))
    # make all pytorch tensors into np arrays, which consume less disk space
    param_history = dict(zip(param_history.keys(),map(lambda x: x.detach().numpy(), param_history.values())))
    if not track_params:
        # save just last values, since that's all we need for incremental inference, and the rest fills too much space
        param_history = dict(zip(param_history.keys(), map(lambda x: x[-1], param_history.values())))
    return losses, lppds, param_history, initcopy, round(time.time() - start)