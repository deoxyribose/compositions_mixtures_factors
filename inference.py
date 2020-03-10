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
import pyro.optim
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.predictive import _predictive
from torch.distributions import constraints
from pyro import distributions as dst
from collections import defaultdict
from functools import wraps

def p_value_of_slope(loss, window, slope_significance):
    if len(loss) < window or any(map(torch.isinf, loss)) or slope_significance == 1:
        return 0
    else:
        recent = loss[-window:]
        return sps.linregress(np.arange(window),recent)[3]

def get_lppd(model, guide, data, init, n_samples = 10000):
    # sample N latent variables from guide
    posterior_samples = _predictive(guide, {}, n_samples, parallel = True, model_args = [data[0:1,:], 1, init])
    # posterior_samples['scale'] is by default a (n_samples,1,D), needs to be (n_samples,D) or we get a Cartesian product
    posterior_samples['scale'] = torch.squeeze(posterior_samples['scale'])
    # condition N models on latent variables
    unconditioned_model = pyro.poutine.uncondition(model)
    pred = pyro.poutine.condition(unconditioned_model, posterior_samples)
    pred_trace = pyro.poutine.trace(pred).get_trace(torch.empty((n_samples,2)), n_samples, init)
    # get log_prob of each of the N conditioned models on test data
    log_probs = pred_trace.nodes['obs']['fn'].log_prob(torch.unsqueeze(test_data, dim=-2))
    # logsumexp over the N models minus log(N) (= divide the probability sum by N)
    # mean over test data
    return (log_probs.logsumexp(dim=1)-np.log(n_samples)).mean()

def inference(model, guide, training_data, test_data, init, n_iter = 10000, window = 500, batch_size = 10, n_mc_samples = 16, learning_rate = 0.1, learning_rate_decay = 0.9999, n_posterior_samples = 800, slope_significance = 0.5, track_params = False):
    pyro.clear_param_store()

    #def per_param_callable(module_name, param_name):
    #    return {"lr": learning_rate, "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html
    conditioned_model = pyro.condition(model, data = {'obs': training_data})
    optim = torch.optim.Adam
    scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': {"lr": learning_rate, "betas": (0.90, 0.999)}, 'gamma': learning_rate_decay})
    #scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': learning_rate_decay})
    if 'Mixture' in model.__repr__():
        elbo = TraceEnum_ELBO(max_plate_nesting=3)
    else:
        elbo = Trace_ELBO()
    svi = SVI(model, guide, scheduler, loss=elbo, num_samples=10)

    # Register hooks to monitor gradient norms.
    losses = []
    lrs = []
    gradient_norms = defaultdict(list)
    loss = svi.step(training_data, batch_size, init)
    param_history = dict({k:v.unsqueeze(0) for k,v in pyro.get_param_store().items()})
    # register gradient hooks for monitoring
    #for name, value in pyro.get_param_store().named_parameters():
    #    value.register_hook(lambda g, name=name: gradient_norms[name].append(g.norm().item()))

    # print current job
    print("Training {} with {}".format(model.__repr__(), init[0][0]))
    # optimize
    i = 0
    raw_batch_size = batch_size
    n_posterior_samples = n_posterior_samples//10
    lppds = []
    with torch.no_grad():
        lppd = get_lppd(model, guide, test_data, init, n_samples=n_posterior_samples)
        lppds.append(-lppd)
        print(lppds[-1])

    while p_value_of_slope(lppds,10, slope_significance) < slope_significance and i < n_iter:
        try:
            loss = svi.step(training_data, batch_size, init)
            if i % window or i <= window:
                print('.', end='')
                scheduler.step()
                #state = scheduler.get_state()['loc_loc']
                #lr = state['base_lrs'][0]*state['gamma']**state['last_epoch']
            else:
                #print('\nSetting lr to {} after {} iterations'.format(lr, i), end='')
                #print('\n', end='')
                #raw_batch_size *= 1.05n_posterior_samplesn_posterior_samples

                n_posterior_samples += 1000
                n_posterior_samples = min(n_posterior_samples, 10000)

                raw_batch_size += 2
                batch_size = min(16,int(raw_batch_size))

                svi.num_samples += 1
                svi.num_samples = min(n_mc_samples,svi.num_samples)

                with torch.no_grad():
                    lppd = get_lppd(model, guide, test_data, init, n_samples=n_posterior_samples)
                    lppds.append(-lppd)
                    print('\n')
                    print(lppds[-1])
                print('\n')
                print('{}/{}'.format(i,n_iter))
                print('\nSetting number of MC samples to {}'.format(svi.num_samples), end='')
                print('\nSetting number of posterior samples to {}'.format(n_posterior_samples), end='')
                print('\nSetting batch size to {}'.format(batch_size), end='')
            losses.append(loss)
            #lrs.append(lr)
            # take one svi step to populate the param store
            # warning: this builds the param_history dict from scratch in every iteration
            param_history = {k:torch.cat([param_history[k],v.unsqueeze(0).detach()],dim=0) for k,v in pyro.get_param_store().items()}
            i += 1
        except KeyboardInterrupt:
            print('\Interrupted by user after {} iterations.\n'.format(i))
            return svi, losses, lppds, param_history, init, gradient_norms
    print('\nConverged in {} iterations.\n'.format(i))
    # make all pytorch tensors into np arrays, which consume less disk space
    param_history = dict(zip(param_history.keys(),map(lambda x: x.detach().numpy(), param_history.values())))
    if not track_params:
        # save just last values, since that's all we need for incremental inference, and the rest fills too much space
        param_history = dict(zip(param_history.keys(), map(lambda x: x[-1], param_history.values())))
    return svi, losses, lppds, param_history, init, gradient_norms