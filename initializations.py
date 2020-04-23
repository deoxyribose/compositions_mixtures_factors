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
from torch.distributions import constraints
from pyro import distributions as dist
from collections import defaultdict
from sklearn.decomposition import PCA
import sys
sys.path.append("..")


def set_uninformative_priors(K,D,prior_std, param_history = None):
    scaleloc = torch.zeros(D)
    scalescale = prior_std*torch.ones(D)
    cov_factor_loc = torch.zeros(K,D)
    cov_factor_scale = prior_std*torch.ones(K,D)
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_random_variational_parameter_init(K,D,prior_std, param_history = None):
    scaleloc = torch.abs(torch.randn(D))#torch.randn(D)
    scalescale = torch.abs(torch.randn(D))
    #scalescale = prior_std*torch.abs(torch.randn(1))
    cov_factor_loc = torch.randn(K,D)
    cov_factor_scale = torch.abs(torch.randn(K,D))
    #cov_factor_scale = prior_std*torch.abs(torch.randn(K,D))
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_PCA_variational_parameter_init(K,D,prior_std,data, param_history = None):
    scaleloc = torch.abs(torch.randn(D))#torch.randn(D)
    scalescale = torch.abs(torch.randn(D))
    #scalescale = prior_std*torch.abs(torch.randn(1))
    tmp = PCA(n_components=K)
    tmp.fit(data)
    cov_factor_loc = torch.tensor(tmp.components_,dtype = torch.float32)
    cov_factor_scale = torch.abs(torch.randn(K,D))
    #cov_factor_scale = prior_std*torch.abs(torch.randn(K,D))
    return K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

def set_incremental_hyperparameter_init(K,D,prior_std, param_history = None):
    print('Initializing hyperparameters in those learnt by previous model.')
    if K == 2:
        prev_posterior_loc = torch.tensor(param_history['cov_factor_loc_hyper_{}'.format(K-1)])
        prev_posterior_scale = torch.tensor(param_history['cov_factor_scale_hyper_{}'.format(K-1)])
    else:
        cov_factor_loc = torch.tensor(param_history['cov_factor_loc_hyper_{}'.format(K-1)])
        cov_factor_scale = torch.tensor(param_history['cov_factor_scale_hyper_{}'.format(K-1)])
        cov_factor_new_loc = torch.tensor(param_history['cov_factor_new_loc_hyper_{}'.format(K-1)])
        cov_factor_new_scale = torch.tensor(param_history['cov_factor_new_scale_hyper_{}'.format(K-1)])

        prev_posterior_loc = torch.cat([cov_factor_loc,torch.unsqueeze(cov_factor_new_loc,dim=0)])
        prev_posterior_scale = torch.cat([cov_factor_scale,torch.unsqueeze(cov_factor_new_scale,dim=0)])
    cov_loc_init = torch.zeros(K,D)
    cov_loc_init[:K-1,:] = prev_posterior_loc
    cov_scale_init = torch.ones(K,D)
    #cov_scale_init = prior_std*torch.ones(K,D)
    cov_scale_init[:K-1,:] = prev_posterior_scale
    return K, torch.tensor(param_history['scale_loc_hyper']),torch.tensor(param_history['scale_scale_hyper']),cov_loc_init,cov_scale_init

def set_incremental_variational_parameter_init(K,D,prior_std, param_history = None):
    print('Initializing variational parameters in those learnt by previous model.')
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

def get_h_and_v_params(K,D,condition = 0, prior_std = 1, data = None, param_history = None):
    assert D > 0
    assert K > 0
    if condition == 0:
        #return set_uninformative_priors(K, D, prior_std), set_random_variational_parameter_init(K, D, prior_std)
        return set_random_variational_parameter_init(K, D, prior_std), set_PCA_variational_parameter_init(K,D,prior_std,data)
    if condition == 1:
        # since incremental variational parameters are the same as the incremental prior parameters
    #    return set_uninformative_priors(K, D, prior_std), set_incremental_variational_parameter_init(K, D, prior_std)
        return set_incremental_hyperparameter_init(K, D, prior_std, param_history), set_incremental_variational_parameter_init(K, D, prior_std, param_history)
    else:
        return set_random_variational_parameter_init(K, D, prior_std), set_random_variational_parameter_init(K, D, prior_std)


def clone_init(init):
    """
    Convenience function for cloning initializations, avoiding continued training on original init
    """
    clone = [[],[]]
    for i,parameter_set in enumerate(init):
        for param in parameter_set:
            if type(param) == torch.Tensor:
                clone[i].append(param.clone().detach())
            else:
                clone[i].append(param)
        clone[i] = tuple(clone[i])
    return tuple(clone)


def get_param_history_of_best_restart(pickled_model):
    print("Loading best restart from {}".format(pickled_model))
    with open(pickled_model, 'rb') as f:
        results = pickle.load(f)
    best_lppd_at_convergence = np.inf
    for result in results:
        _,lppds,param_history,_,_ = result
        mean_lppd_at_convergence = sum(lppds[-10:])/10
        if mean_lppd_at_convergence < best_lppd_at_convergence:
            best_lppd_at_convergence = mean_lppd_at_convergence
            best_param_history = param_history
    return best_param_history