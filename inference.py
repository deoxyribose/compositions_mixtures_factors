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
import operator
sys.path.append("..")
from initializations import *

def p_value_of_slope(loss, window, slope_significance):
    if len(loss) < window or any(map(torch.isinf, loss)) or slope_significance == 1:
        return 0
    else:
        recent = loss[-window:]
        return sps.linregress(np.arange(window),recent)[3]

def inference(Model, training_data, test_data, config = None):
    '''
    A wrapper function calling Pyro's SVI step with settings given in config.
    Records telemetry including elbo loss, mean negative log likelihood on held out data, gradient norms and parameter history during training.
    If config includes telemetry from a previous inference run, inference continues from that run.
    If slope_significance is set to a value less than 1, training halts when the mean negative log likelihood converges.
    Convergence is estimated by linear regression in a moving window of size convergence_window when p(slope=estimate|true_slope=0) < slope_significance.

    Default config is 
    config = dict(
            n_iter = 1000,
            learning_rate = 0.1, 
            beta1 = 0.9,
            beta2 = 0.999,
            learning_rate_decay = 1., # no decay by default
            batch_size = 32, 
            n_elbo_particles = 32, 
            n_posterior_samples = 1024,
            window = 500,
            convergence_window = 30,
            slope_significance = 0.1,
            track_params = False,
            monitor_gradients = False,
            optimizer_state = None,
            param_store_state = None,
        )
    '''
    #initcopy = clone_init(init)
    if config is None:
        config = dict(
                n_iter = 1000,
                learning_rate = 0.1, 
                beta1 = 0.9,
                beta2 = 0.999,
                learning_rate_decay = 1., # no decay by default
                batch_size = 32, 
                n_elbo_particles = 32, 
                n_posterior_samples = 1024,
                window = 500,
                convergence_window = 30,
                slope_significance = 0.1,
                track_params = False,
                monitor_gradients = False,
                telemetry = None,
            )
    #def per_param_callable(module_name, param_name):
    #    return {"lr": config['learning_rate'], "betas": (0.90, 0.999)} # from http://pyro.ai/examples/svi_part_i.html
    model = Model.model
    guide = Model.guide

    optim = pyro.optim.Adam({"lr": config['learning_rate'], "betas": (config['beta1'], config['beta2'])})
    
    # if there is previous telemetry in the config from an interrupted inference run
    # restore the state of that inference and continue training
    if config['telemetry']:
        pyro.clear_param_store()
        print('Continuing from previous inference run.')
        telemetry = config['telemetry']
        optim.set_state(telemetry['optimizer_state'])
        pyro.get_param_store().set_state(telemetry['param_store_state'])
        i = len(telemetry['loss'])
        config['n_iter'] += i
    else:
        pyro.clear_param_store()
        telemetry = dict()
        telemetry['gradient_norms'] = defaultdict(list)
        telemetry['loss'] = []
        telemetry['MNLL'] = []
        telemetry['training_duration'] = 0
        # call model and guide to populate param store
        #model(training_data, config['batch_size'], init)
        #guide(training_data, config['batch_size'], init)
        Model.batch_size = config['batch_size']
        model(training_data)
        guide(training_data)
        telemetry['param_history'] = dict({k:v.unsqueeze(0) for k,v in pyro.get_param_store().items()})
        i = 0
    # Learning rate schedulers
    # Haven't found a way to get and set its state for checkpointing
    #optim = torch.optim.Adam
    #scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': {"lr": config['learning_rate'], "betas": (beta1, beta2)}, 'gamma': config['learning_rate_decay']})
    #scheduler = pyro.optim.ExponentialLR({'optimizer': optim, 'optim_args': per_param_callable, 'gamma': config['learning_rate_decay']})
    
    max_plate_nesting = _guess_max_plate_nesting(model,(training_data,),{})
    print("Guessed that model has max {} nested plates.".format(max_plate_nesting)) 
    if 'Mixture' in model.__repr__():
        elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
    else:
        elbo = Trace_ELBO(max_plate_nesting=max_plate_nesting, num_particles=config['n_elbo_particles'], vectorize_particles=True)
    #svi = SVI(model, guide, scheduler, loss=elbo)
    svi = SVI(model, guide, optim, loss=elbo)

    if config['monitor_gradients']:
        # register gradient hooks for monitoring
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: telemetry['gradient_norms'][name].append(g.norm().item()))
    start = time.time()
    
#    with torch.no_grad():
#        mnll = compute_mnll(model, guide, test_data, init, n_samples=config['n_posterior_samples'])
#        telemetry['MNLL'].append(-mnll)
#        print("NLL at init is {}".format(mnlls[-1]))

    while p_value_of_slope(telemetry['MNLL'],config['convergence_window'], config['slope_significance']) < config['slope_significance'] and i < config['n_iter']:
        try:
            loss = svi.step(training_data)
            telemetry['loss'].append(loss)
            if i % config['window'] or i <= config['window']:
                print('.', end='')
                #scheduler.step()
            else:
                with torch.no_grad():
                    #mnll = compute_mnll(model, guide, test_data, n_samples=config['n_posterior_samples'])
                    telemetry['MNLL'].append(-Model.mnll(test_data, config['n_posterior_samples']))
                    print('\n')
                    print("NLL after {}/{} iterations is {}".format(i,config['n_iter'], telemetry['MNLL'][-1]))
                print('\n')
                #print('\nSetting number of posterior samples to {}'.format(config['n_posterior_samples']), end='')
                #print('\nSetting batch size to {}'.format(config['batch_size']), end='')
            if config['track_params']:
                telemetry['param_history'] = {k:torch.cat([telemetry['param_history'][k],v.unsqueeze(0).detach()],dim=0) for k,v in pyro.get_param_store().items()}
            i += 1
        except KeyboardInterrupt:
            print('\Interrupted by user after {} iterations.\n'.format(i))
            params = {k:v.detach() for k,v in pyro.get_param_store().items()}
            Model.params = params
            telemetry['training_duration'] += round(time.time() - start)
            telemetry['optimizer_state'] = optim.get_state()
            telemetry['param_store_state'] = pyro.get_param_store().get_state()
            return telemetry
    print('\nConverged in {} iterations.\n'.format(i))

    # make all pytorch tensors into np arrays, which consume less disk space
    #param_history = dict(zip(param_history.keys(),map(lambda x: x.detach().numpy(), param_history.values())))
    
    params = {k:v.detach() for k,v in pyro.get_param_store().items()}
    Model.params = params
    telemetry['training_duration'] += round(time.time() - start)
    telemetry['optimizer_state'] = optim.get_state()
    telemetry['param_store_state'] = pyro.get_param_store().get_state()
    return telemetry

def successive_halving(inference_args, init_func, n_seeds = 2**3, n_iters_in_interval = 100):
    """
    Trains a number of models from n_seeds different inits generated by init_func,, 
    pruning the worst performing half of models every n_iters_in_interval iterations,
    until only one model is left.
    Returns best inference and inference results from all models.
    """
    model, guide, data, test_data, init, config = inference_args
    original_seeds = range(42,42+n_seeds)
    all_inference_results = {seed:None for seed in original_seeds}
    config['n_iter'] = n_iters_in_interval # run all seeds for 100 iters at a time
    config['telemetry'] = None
    config['slope_significance'] = 1
    # find how many halvings until just one is left
    n_intervals = np.int(np.ceil(np.log(n_seeds)/np.log(2)))
    seeds = list(original_seeds).copy()
    for interval in range(n_intervals+1):
        # prune bad seeds
        if interval:
            performance = {seed:all_inference_results[seed][2]['MNLL'][-1] for seed in seeds}
            sorted_performance = sorted(performance.items(), key=operator.itemgetter(1))
            seeds = dict(sorted_performance[:(len(sorted_performance)//2)]).keys()
        for seed in seeds:
            if interval:
                config['telemetry'] = all_inference_results[seed][2]
                config['n_iter'] = n_iters_in_interval
            else:
                pyro.set_rng_seed(seed)
                init = init_func()
            inference_results = inference(model, guide, data, test_data, init, config)
            all_inference_results[seed] = inference_results
    winner_seed = list(seeds)[0]
    best_inference_results = all_inference_results[winner_seed]
    return best_inference_results, all_inference_results


#def get_mnll(model, guide, data, init, n_samples = 1000, verbose = True):
#    """
#    Parallel implementation, where samples from the posterior are batched. Doesn't work for big datasets due to low_rank_multivariate_normal's log_prob having to matmul large matrices
#    Slower than looping over posterior samples as in compute_mnll
#    """
#    # sample N latent variables from guide
#    posterior_samples = _predictive(guide, {}, n_samples, parallel = True, model_args = [data[0:1,:], 1, init])
#    # posterior_samples['scale'] is by default a (n_samples,1,D), needs to be (n_samples,D) or we get a Cartesian product
#    posterior_samples['scale'] = torch.squeeze(posterior_samples['scale'])
#    unconditioned_model = pyro.poutine.uncondition(model)
#    pred = pyro.poutine.condition(unconditioned_model, posterior_samples)
#    pred_trace = pyro.poutine.trace(pred).get_trace(torch.empty((n_samples,data.shape[1])), n_samples, init)
#    log_probs = torch.empty((data.shape[0],n_samples))
#    n_partitions = 50
#    for i in range(n_partitions):
#        idx0 = i*data.shape[0]//n_partitions
#        idx1 = (i+1)*data.shape[0]//n_partitions
#        log_probs[idx0:idx1,:] = pred_trace.nodes['obs']['fn'].log_prob(torch.unsqueeze(data[idx0:idx1,:], dim=-2))
#    return (log_probs.logsumexp(dim=1)-np.log(n_samples)).mean()
#
#def compute_mnll(model, guide, data, n_samples = 1000):
#    '''
#    Estimate mean negative log likelihood on data samples using samples from variational posterior.
#    E_{x_j~X}[log E_{z_i~q}[p(x_j|z_i)]]
#    '''
#    unconditioned_model = pyro.poutine.uncondition(model)
#    dummy_obs = data[0:1,:]
#    log_probs = torch.zeros((data.shape[0],n_samples))
#    for model_idx in range(n_samples):
#        guide_trace = pyro.poutine.trace(guide).get_trace(dummy_obs, 1)
#        blockreplay = pyro.poutine.block(fn = pyro.poutine.replay(unconditioned_model, guide_trace),expose=['obs'])
#        posterior_predictive = pyro.poutine.trace(blockreplay).get_trace(dummy_obs, 1)
#        log_probs[:,model_idx] = posterior_predictive.nodes['obs']['fn'].log_prob(data)
#    return (torch.logsumexp(log_probs,dim=1)-np.log(np.float(n_samples))).mean()
##
#def compute_mnll(model, guide, data, init, n_samples = 1000):
#    '''
#    Estimate mean negative log likelihood on data samples using samples from variational posterior.
#    E_{x_j~X}[log E_{z_i~q}[p(x_j|z_i)]]
#    '''
#    unconditioned_model = pyro.poutine.uncondition(model)
#    dummy_obs = data[0:1,:]
#    log_probs = torch.zeros((data.shape[0],n_samples))
#    for model_idx in range(n_samples):
#        guide_trace = pyro.poutine.trace(guide).get_trace(dummy_obs, 1, init)
#        blockreplay = pyro.poutine.block(fn = pyro.poutine.replay(unconditioned_model, guide_trace),expose=['obs'])
#        posterior_predictive = pyro.poutine.trace(blockreplay).get_trace(dummy_obs, 1, init)
#        log_probs[:,model_idx] = posterior_predictive.nodes['obs']['fn'].log_prob(data)
#    return (torch.logsumexp(log_probs,dim=1)-np.log(np.float(n_samples))).mean()
