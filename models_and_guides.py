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
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, Predictive
from torch.distributions import constraints
from pyro import distributions as dist
from collections import defaultdict
import sys
sys.path.append("..")
from initializations import *

class Model():
    def __init__(self, X, batch_size, _id):
        self.X = X
        self.N, self.D = self.X.shape
        self._id = _id
        self.batch_size = batch_size

    def initialize_parameters(self):
        return get_random_init(self.param_shapes_and_support)

    def model(self, X):
        raise NotImplementedError
        
    def unconditioned_model(self, *args):
        return pyro.poutine.uncondition(self.model)(*args)
    
    def guide(self, X):
        raise NotImplementedError
        
    def mnll(self, data, n_samples):
        '''
        Estimate mean negative log likelihood on observed data given samples from variational posterior.
        E_{x_j~X}[log E_{z_i~q}[p(x_j|z_i)]]

        WARNING: Must only be called inside a with torch.no_grad():, 
        # otherwise pytorch will try to diff through this
        '''
        original_batch_size = self.batch_size
        self.batch_size = 1
        dummy_obs = data[0:1,:]
        log_probs = torch.zeros((data.shape[0],n_samples))
        for model_idx in range(n_samples):
            guide_trace = pyro.poutine.trace(self.guide).get_trace(dummy_obs)
            blockreplay = pyro.poutine.block(fn = pyro.poutine.replay(self.unconditioned_model, guide_trace),expose=['obs'])
            posterior_predictive = pyro.poutine.trace(blockreplay).get_trace(dummy_obs)
            log_probs[:,model_idx] = posterior_predictive.nodes['obs']['fn'].log_prob(data)
        self.batch_size = original_batch_size
        return (torch.logsumexp(log_probs,dim=1)-np.log(np.float(n_samples))).mean()
    
    def sample_posterior_predictive(self, n_samples):
        original_batch_size = self.batch_size
        self.batch_size = 1
        self.posterior_predictive = Predictive(self.unconditioned_model,guide=self.guide,num_samples=n_samples)
        self.posterior_samples = self.posterior_predictive(self.X)
        self.posterior_predictive_samples = self.posterior_samples['obs'].squeeze().detach()
        self.batch_size = original_batch_size

#
#class ZeroMeanFactor(Model):
#    def __init__(self, X, K, batch_size, _id):
#        super(ZeroMeanFactor, self).__init__(X, batch_size, _id)
#        self.K = K
#        self.param_shapes_and_support = self.get_param_shapes_and_support()
#        self.param_init = self.initialize_parameters()
#
#    def get_param_shapes_and_support(self, _id = None):
#        if _id == None:
#            _id = self._id
#        return {f'cov_diag_prior_loc_init_{_id}': ((self.D,), constraints.real),
#                f'cov_diag_prior_scale_init_{_id}': ((self.D,), constraints.positive),
#                f'cov_factor_prior_loc_init_{_id}': ((self.K, self.D), constraints.real),
#                f'cov_factor_prior_scale_init_{_id}': ((self.K, self.D), constraints.positive),
#                f'cov_diag_loc_init_{_id}': ((self.D,), constraints.real),
#                f'cov_diag_scale_init_{_id}': ((self.D,), constraints.positive),
#                f'cov_factor_loc_init_{_id}': ((self.K, self.D), constraints.real),
#                f'cov_factor_scale_init_{_id}': ((self.K, self.D), constraints.positive)}
#
#    def model(self, X):
#        _id = self._id
#        K = self.K
#        #N, D = self.X.shape
#        N, D = X.shape
#        cov_diag_locinit = self.param_init[f'cov_diag_prior_loc_init_{_id}']
#        cov_diag_scale_init = self.param_init[f'cov_diag_prior_scale_init_{_id}']
#        cov_factor_loc_init = self.param_init[f'cov_factor_prior_loc_init_{_id}']
#        cov_factor_scale_init = self.param_init[f'cov_factor_prior_scale_init_{_id}']
#        with pyro.plate(f'D_{_id}', D):
#            cov_diag_loc = pyro.param(f'cov_diag_prior_loc_{_id}', cov_diag_locinit)
#            cov_diag_scale = pyro.param(f'cov_diag_prior_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
#            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale))
#            cov_factor = None
#            if K > 1:
#                with pyro.plate(f'K_{_id}', K-1):
#                    cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init[:K-1,:])
#                    cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
#                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc,cov_factor_scale))
#                cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init[-1,:])
#                cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init[-1,:], constraint=constraints.positive)
#                cov_factor_new = pyro.sample(f'cov_factor_new_{_id}', dist.Normal(cov_factor_loc[-1,:],cov_factor_scale[-1,:]))
#                #cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
#                if cov_factor_new.dim() == cov_factor.dim():
#                    cov_factor = torch.cat([cov_factor, cov_factor_new], dim=-2)
#                    #cov_factor = torch.cat([cov_factor, cov_factor_new], dim=1)
#                else:
#                    cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=-2)], dim=-2)
#            else:
#                with pyro.plate(f'K_{_id}', K):
#                    cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init)
#                    cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init, constraint=constraints.positive)
#                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc,cov_factor_scale))
#            cov_factor = cov_factor.transpose(-2,-1)
#            loc = torch.zeros(D)
#        with pyro.plate(f'N_{_id}', size=N, subsample_size=self.batch_size) as ind:
#            X = pyro.sample('obs', dist.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
#        return X
    #
#    def guide(self, X):
#        _id = self._id
#        N, D = X.shape
#        K = self.K
#        cov_diag_loc_init = self.param_init[f'cov_diag_loc_init_{_id}']
#        cov_diag_scale_init = self.param_init[f'cov_diag_scale_init_{_id}']
#        cov_factor_loc_init = self.param_init[f'cov_factor_loc_init_{_id}']
#        cov_factor_scale_init = self.param_init[f'cov_factor_scale_init_{_id}']
#        with pyro.plate(f'D_{_id}', D, dim=-1):
#            cov_diag_loc = pyro.param(f'cov_diag_loc_{_id}', cov_diag_loc_init)
#            cov_diag_scale = pyro.param(f'cov_diag_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
#            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale))
#            cov_diag = cov_diag*torch.ones(D)
#            # sample variables
#            cov_factor = None
#            if K > 1:
#                with pyro.plate(f'K_{_id}', K-1, dim=-2):
#                    cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init[:K-1,:])
#                    cov_factor_scale = pyro.param(f'cov_factor_scale_{_id}', cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
#                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc, cov_factor_scale))
#                cov_factor_new_loc = pyro.param(f'cov_factor_new_loc_{_id}', cov_factor_loc_init[-1,:])
#                cov_factor_new_scale = pyro.param(f'cov_factor_new_scale_{_id}', cov_factor_scale_init[-1,:], constraint=constraints.positive)
#                cov_factor_new = pyro.sample(f'cov_factor_new_{_id}', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
#                # when using pyro.infer.Predictive, cov_factor_new is somehow sampled as 2-d tensors instead of 1-d
#                if cov_factor_new.dim() == cov_factor.dim():
#                    cov_factor = torch.cat([cov_factor, cov_factor_new], dim=-2)
#                else:
#                    cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=-2)], dim=-2)
#            else:
#                with pyro.plate(f'K_{_id}', K):
#                    cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init)
#                    cov_factor_scale = pyro.param(f'cov_factor_scale_{_id}', cov_factor_scale_init, constraint=constraints.positive)
#                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc, cov_factor_scale))
#            cov_factor = cov_factor.transpose(-2,-1)
#        return cov_factor, cov_diag

class ZeroMeanFactor(Model):
    def __init__(self, X, K, batch_size, _id):
        super(ZeroMeanFactor, self).__init__(X, batch_size, _id)
        self.K = K
        self.param_shapes_and_support = self.get_param_shapes_and_support()
        self.param_init = self.initialize_parameters()

    def get_param_shapes_and_support(self, _id = None):
        if _id == None:
            _id = self._id
        return {f'cov_diag_prior_loc_init_{_id}': ((self.D,), constraints.real),
                f'cov_diag_prior_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_factor_prior_loc_init_{_id}': ((self.K, self.D), constraints.real),
                f'cov_factor_prior_scale_init_{_id}': ((self.K, self.D), constraints.positive),
                f'cov_diag_loc_init_{_id}': ((self.D,), constraints.real),
                f'cov_diag_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_factor_loc_init_{_id}': ((self.K, self.D), constraints.real),
                f'cov_factor_scale_init_{_id}': ((self.K, self.D), constraints.positive)}

    def model(self, X):
        _id = self._id
        K = self.K
        #N, D = self.X.shape
        N, D = X.shape
        cov_diag_locinit = self.param_init[f'cov_diag_prior_loc_init_{_id}']
        cov_diag_scale_init = self.param_init[f'cov_diag_prior_scale_init_{_id}']
        cov_factor_loc_init = self.param_init[f'cov_factor_prior_loc_init_{_id}']
        cov_factor_scale_init = self.param_init[f'cov_factor_prior_scale_init_{_id}']
        with pyro.plate(f'D_{_id}', D):
            cov_diag_loc = pyro.param(f'cov_diag_prior_loc_{_id}', cov_diag_locinit)
            cov_diag_scale = pyro.param(f'cov_diag_prior_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale))
            jitter = 1e-05
            cov_diag = cov_diag + jitter
            with pyro.plate(f'K_{_id}', K):
                cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init)
                cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc,cov_factor_scale))
            cov_factor = cov_factor.transpose(-2,-1)
            loc = torch.zeros(D)
        with pyro.plate(f'N_{_id}', size=N, subsample_size=self.batch_size) as ind:
            X = pyro.sample('obs', dist.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
        return X
    
    def guide(self, X):
        _id = self._id
        N, D = X.shape
        K = self.K
        cov_diag_loc_init = self.param_init[f'cov_diag_loc_init_{_id}']
        cov_diag_scale_init = self.param_init[f'cov_diag_scale_init_{_id}']
        cov_factor_loc_init = self.param_init[f'cov_factor_loc_init_{_id}']
        cov_factor_scale_init = self.param_init[f'cov_factor_scale_init_{_id}']
        with pyro.plate(f'D_{_id}', D, dim=-1):
            cov_diag_loc = pyro.param(f'cov_diag_loc_{_id}', cov_diag_loc_init)
            cov_diag_scale = pyro.param(f'cov_diag_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale))
            jitter = 1e-05
            cov_diag = cov_diag + jitter
            # sample variables
            with pyro.plate(f'K_{_id}', K, dim=-2):
                cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init)
                cov_factor_scale = pyro.param(f'cov_factor_scale_{_id}', cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor = cov_factor.transpose(-2,-1)
        return cov_factor, cov_diag

class Factor(Model):
    def __init__(self, X, K, batch_size, _id):
        super(Factor, self).__init__(X, batch_size, _id)
        self.K = K
        self.param_shapes_and_support = self.get_param_shapes_and_support()
        self.param_init = self.initialize_parameters()

    def get_param_shapes_and_support(self, _id = None):
        if _id == None:
            _id = self._id
        return {f'loc_prior_loc_init_{_id}': ((self.D,), constraints.real),
                f'loc_prior_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_diag_prior_loc_init_{_id}': ((self.D,), constraints.real),
                f'cov_diag_prior_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_factor_prior_loc_init_{_id}': ((self.K, self.D), constraints.real),
                f'cov_factor_prior_scale_init_{_id}': ((self.K, self.D), constraints.positive),
                f'cov_diag_loc_init_{_id}': ((self.D,), constraints.real),
                f'cov_diag_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_factor_loc_init_{_id}': ((self.K, self.D), constraints.real),
                f'cov_factor_scale_init_{_id}': ((self.K, self.D), constraints.positive)}

    def model(self, X):
        _id = self._id
        K = self.K
        N, D = X.shape
        loc_locinit = self.param_init[f'loc_prior_loc_init_{_id}']
        loc_scaleinit = self.param_init[f'loc_prior_scale_init_{_id}']
        cov_diag_loc_init = self.param_init[f'cov_diag_prior_loc_init_{_id}']
        cov_diag_scale_init = self.param_init[f'cov_diag_prior_scale_init_{_id}']
        cov_factor_loc_init = self.param_init[f'cov_factor_prior_loc_init_{_id}']
        cov_factor_scale_init = self.param_init[f'cov_factor_prior_scale_init_{_id}']
        with pyro.plate(f'D_{_id}', D):
            loc_loc = pyro.param(f'loc_loc_prior_{_id}', loc_locinit)
            loc_scale = pyro.param(f'loc_scale_prior_{_id}', loc_scaleinit, constraint=constraints.positive)
            loc = pyro.sample(f'loc_{_id}', dist.LogNormal(loc_loc, loc_scale))

            cov_diag_loc = pyro.param(f'cov_diag_prior_loc_{_id}', cov_diag_loc_init)
            cov_diag_scale = pyro.param(f'cov_diag_prior_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale))
            jitter = 1e-05
            cov_diag = cov_diag + jitter
            cov_factor = None
            if K > 1:
                with pyro.plate(f'K_{_id}', K-1):
                    cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init[:K-1,:])
                    cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc,cov_factor_scale))
                cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init[-1,:])
                cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init[-1,:], constraint=constraints.positive)
                cov_factor_new = pyro.sample(f'cov_factor_new_{_id}', dist.Normal(cov_factor_loc_init[-1,:],cov_factor_scale_init[-1,:]))
                #cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
                if cov_factor_new.dim() == cov_factor.dim():
                    cov_factor = torch.cat([cov_factor, cov_factor_new], dim=-2)
                    #cov_factor = torch.cat([cov_factor, cov_factor_new], dim=1)
                else:
                    cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=-2)], dim=-2)
            else:
                with pyro.plate(f'K_{_id}', K):
                    cov_factor_loc = pyro.param(f'cov_factor_prior_loc_{_id}', cov_factor_loc_init)
                    cov_factor_scale = pyro.param(f'cov_factor_prior_scale_{_id}', cov_factor_scale_init, constraint=constraints.positive)
                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc,cov_factor_scale))
            cov_factor = cov_factor.transpose(-2,-1)
        with pyro.plate(f'N_{_id}', size=N, subsample_size=self.batch_size) as ind:
            X = pyro.sample('obs', dist.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
        return X
    
    def guide(self, X):
        _id = self._id
        K = self.K
        N, D = X.shape
        loc_locinit = self.param_init[f'loc_loc_init_{_id}']
        loc_scaleinit = self.param_init[f'loc_scale_init_{_id}']
        cov_diag_loc_init = self.param_init[f'cov_diag_loc_init_{_id}']
        cov_diag_scale_init = self.param_init[f'cov_diag_scale_init_{_id}']
        cov_factor_loc = self.param_init[f'cov_factor_loc_init_{_id}']
        cov_factor_scale = self.param_init[f'cov_factor_scale_init_{_id}']
        with pyro.plate(f'D_{_id}', D, dim=-1):
            loc_loc = pyro.param(f'loc_loc_{_id}', loc_locinit)
            loc_scale = pyro.param(f'loc_scale_{_id}', loc_scaleinit, constraint=constraints.positive)
            loc = pyro.sample(f'loc_{_id}', dist.LogNormal(loc_loc, loc_scale))

            cov_diag_loc = pyro.param(f'cov_diag_loc_{_id}', cov_diag_loc_init)
            cov_diag_scale = pyro.param(f'cov_diag_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
            cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale))
            cov_diag = cov_diag*torch.ones(D)
            jitter = 1e-05
            cov_diag = cov_diag + jitter
            # sample variables
            cov_factor = None
            if K > 1:
                with pyro.plate(f'K_{_id}', K-1, dim=-2):
                    cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init[:K-1,:])
                    cov_factor_scale = pyro.param(f'cov_factor_scale_{_id}', cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                    cov_factor = pyro.sample(f'cov_factor{_id}', dist.Normal(cov_factor_loc, cov_factor_scale))
                cov_factor_new_loc = pyro.param(f'cov_factor_new_loc_{_id}', cov_factor_loc_init[-1,:])
                cov_factor_new_scale = pyro.param(f'cov_factor_new_scale_{_id}', cov_factor_scale_init[-1,:], constraint=constraints.positive)
                cov_factor_new = pyro.sample(f'cov_factor_new{_id}', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
                # when using pyro.infer.Predictive, cov_factor_new is somehow sampled as 2-d tensors instead of 1-d
                if cov_factor_new.dim() == cov_factor.dim():
                    cov_factor = torch.cat([cov_factor, cov_factor_new], dim=-2)
                else:
                    cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=-2)], dim=-2)
            else:
                with pyro.plate(f'K_{_id}', K):
                    cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init)
                    cov_factor_scale = pyro.param(f'cov_factor_scale_{_id}', cov_factor_scale_init, constraint=constraints.positive)
                    cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor = cov_factor.transpose(-2,-1)
        return cov_factor, cov_diag

class ZeroMeanFactorARD(Model):
    def __init__(self, X, batch_size, _id):
        super(ZeroMeanFactorARD, self).__init__(X, batch_size, _id)
        self.param_shapes_and_support = self.get_param_shapes_and_support()
        self.param_init = self.initialize_parameters()

    def get_param_shapes_and_support(self, _id = None):
        if _id == None:
            _id = self._id
        return {f'global_shrinkage_prior_scale_init_{_id}': ((1,), constraints.positive),
                f'cov_diag_prior_loc_init_{_id}': ((self.D,), constraints.real),
                f'cov_diag_prior_scale_init_{_id}': ((self.D,), constraints.positive),
                f'global_shrinkage_loc_init_{_id}': ((1,), constraints.real),
                f'global_shrinkage_scale_init_{_id}': ((1,), constraints.positive),
                f'local_shrinkage_loc_init_{_id}': ((self.D,), constraints.real),
                f'local_shrinkage_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_diag_loc_init_{_id}': ((self.D,), constraints.real),
                f'cov_diag_scale_init_{_id}': ((self.D,), constraints.positive),
                f'cov_factor_loc_init_{_id}': ((self.D, self.D), constraints.real)}

    def model(self, X):
        _id = self._id
        N, D = X.shape
        global_shrinkage_prior_scale_init = self.param_init[f'global_shrinkage_prior_scale_init_{_id}']
        cov_diag_prior_loc_init = self.param_init[f'cov_diag_prior_loc_init_{_id}']
        cov_diag_prior_scale_init = self.param_init[f'cov_diag_prior_scale_init_{_id}']


        global_shrinkage_prior_scale = pyro.param(f'global_shrinkage_scale_prior_{_id}', global_shrinkage_prior_scale_init, constraint=constraints.positive)
        tau = pyro.sample(f'global_shrinkage_{_id}', dist.HalfNormal(global_shrinkage_prior_scale))
        
        b = pyro.sample('b', dist.InverseGamma(0.5,1./torch.ones(D)**2).to_event(1))
        lambdasquared = pyro.sample(f'local_shrinkage_{_id}', dist.InverseGamma(0.5,1./b).to_event(1))
        
        cov_diag_loc = pyro.param(f'cov_diag_prior_loc_{_id}', cov_diag_prior_loc_init)
        cov_diag_scale = pyro.param(f'cov_diag_prior_scale_{_id}', cov_diag_prior_scale_init, constraint=constraints.positive)
        cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale).to_event(1))
        #cov_diag = cov_diag*torch.ones(D)
        jitter = 1e-05
        cov_diag = cov_diag + jitter
        
        lambdasquared = lambdasquared.squeeze()
        if lambdasquared.dim() == 1:
            # outer product
            cov_factor_scale = torch.ger(torch.sqrt(lambdasquared),tau.repeat((tau.dim()-1)*(1,)+(D,)))
        else:
            # batch outer product
            cov_factor_scale = torch.einsum('bp, br->bpr', torch.sqrt(lambdasquared),tau.repeat((tau.dim()-1)*(1,)+(D,)))
        cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(0., cov_factor_scale).to_event(2))
        cov_factor = cov_factor.transpose(-2,-1)
        with pyro.plate(f'N_{_id}', size=N, subsample_size=self.batch_size, dim=-1) as ind:
            X = pyro.sample('obs', dist.LowRankMultivariateNormal(torch.zeros(D), cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
        return X

    def guide(self, X):
        _id = self._id
        N, D = X.shape
        global_shrinkage_loc_init = self.param_init[f'global_shrinkage_loc_init_{_id}']
        global_shrinkage_scale_init = self.param_init[f'global_shrinkage_scale_init_{_id}']
        local_shrinkage_loc_init = self.param_init[f'local_shrinkage_loc_init_{_id}']
        local_shrinkage_scale_init = self.param_init[f'local_shrinkage_scale_init_{_id}']
        cov_diag_loc_init = self.param_init[f'cov_diag_loc_init_{_id}']
        cov_diag_scale_init = self.param_init[f'cov_diag_scale_init_{_id}']
        cov_factor_loc_init = self.param_init[f'cov_factor_loc_init_{_id}']

        global_shrinkage_loc = pyro.param(f'global_shrinkage_loc_{_id}', global_shrinkage_loc_init)
        global_shrinkage_scale = pyro.param(f'global_shrinkage_scale_{_id}', global_shrinkage_scale_init, constraint=constraints.positive)
        tau = pyro.sample(f'global_shrinkage_{_id}', dist.LogNormal(global_shrinkage_loc,global_shrinkage_scale))

        b = pyro.sample('b', dist.InverseGamma(0.5,1./torch.ones(D)**2).to_event(1))
        local_shrinkage_loc = pyro.param(f'local_shrinkage_loc_{_id}', local_shrinkage_loc_init)
        local_shrinkage_scale = pyro.param(f'local_shrinkage_scale_{_id}', local_shrinkage_scale_init, constraint=constraints.positive)
        lambdasquared = pyro.sample(f'local_shrinkage_{_id}', dist.LogNormal(local_shrinkage_loc,local_shrinkage_scale).to_event(1))

        cov_diag_loc = pyro.param(f'cov_diag_loc_{_id}', cov_diag_loc_init)
        cov_diag_scale = pyro.param(f'cov_diag_scale_{_id}', cov_diag_scale_init, constraint=constraints.positive)
        cov_diag = pyro.sample(f'cov_diag_{_id}', dist.LogNormal(cov_diag_loc, cov_diag_scale).to_event(1))
        cov_diag = cov_diag*torch.ones(D)

        lambdasquared = lambdasquared.squeeze()
        if lambdasquared.dim() == 1:
            cov_factor_scale = torch.ger(torch.sqrt(lambdasquared),tau.repeat((tau.dim()-1)*(1,)+(D,)))
        else:
            cov_factor_scale = torch.einsum('bp, br->bpr', torch.sqrt(lambdasquared),tau.repeat((tau.dim()-1)*(1,)+(D,)))
        cov_factor_loc = pyro.param(f'cov_factor_loc_{_id}', cov_factor_loc_init)
        cov_factor = pyro.sample(f'cov_factor_{_id}', dist.Normal(cov_factor_loc, cov_factor_scale).to_event(2))
        return tau, lambdasquared, cov_factor, cov_diag

def independentGaussian(X, batch_size, prior_parameters):
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale = prior_parameters[0]
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dist.Normal(locloc,locscale))
        scale = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
    with pyro.plate('N', N, subsample_size=batch_size) as ind:
        # we want to treat the D univariate Normals as one multivariate, so we use to_event
        X = pyro.sample('obs', dist.Normal(loc,scale).to_event(1), obs=X.index_select(0, ind))
    return X

def independentGaussianGuide(X, batch_size, variational_parameter_initialization):
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale = variational_parameter_initialization[1]
    locloc = pyro.param('loc_loc', locloc)
    locscale = pyro.param('loc_scale', locscale, constraint=constraints.positive)
    scaleloc = pyro.param('scale_loc', scaleloc)
    scalescale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dist.Normal(locloc,locscale))
        scale = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
    return loc, scale


def incrementalPpca(X, batch_size, prior_parameters):
    N, D = X.shape
    prior_parameters,_ = prior_parameters
    K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = prior_parameters
    cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D):
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1):
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc[:K-1,:],cov_factor_scale[:K-1,:]))
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_loc[-1,:],cov_factor_scale[-1,:]))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(torch.zeros(D), cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X

def incrementalPpcaGuide(X, batch_size, variational_parameter_initialization):
    N, D = X.shape
    _,variational_parameter_initialization = variational_parameter_initialization
    K, scaleloc, scalescale, cov_factor_loc_init, cov_factor_scale_init = variational_parameter_initialization
    cov_diag_loc = pyro.param('scale_loc', scaleloc)
    cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
    cov_diag = pyro.sample('scale', dist.LogNormal(cov_diag_loc, cov_diag_scale))
    cov_diag = cov_diag*torch.ones(D)
    with pyro.plate('D', D, dim=-1):
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1, dim=-2):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init[:K-1,:])
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor_new_loc = pyro.param('cov_factor_new_loc_{}'.format(K), cov_factor_loc_init[-1,:])
            cov_factor_new_scale = pyro.param('cov_factor_new_scale_{}'.format(K), cov_factor_scale_init[-1,:], constraint=constraints.positive)
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init)
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    return cov_factor, cov_diag


def factor(X, batch_size, prior_parameters):
    """
    Parameters are K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale
    """
    N, D = X.shape
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = prior_parameters[0]
    with pyro.plate('D', D):
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        loc = pyro.sample('loc', dist.Normal(locloc, locscale))
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1):
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc[:K-1,:],cov_factor_scale[:K-1,:]))
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_loc[-1,:],cov_factor_scale[-1,:]))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X

def factorGuide(X, batch_size, variational_parameter_initialization):
    N, D = X.shape
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc_init, cov_factor_scale_init = variational_parameter_initialization[1]
    with pyro.plate('D', D, dim=-1):
        cov_diag_loc = pyro.param('scale_loc', scaleloc)
        cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(cov_diag_loc, cov_diag_scale))
        cov_diag = cov_diag*torch.ones(D)
        loc_loc = pyro.param('loc_loc', locloc)
        loc_scale = pyro.param('loc_scale', locscale, constraint=constraints.positive)
        # sample variables
        loc = pyro.sample('loc', dist.Normal(loc_loc,loc_scale))
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1, dim=-2):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init[:K-1,:])
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor_new_loc = pyro.param('cov_factor_new_loc_{}'.format(K), cov_factor_loc_init[-1,:])
            cov_factor_new_scale = pyro.param('cov_factor_new_scale_{}'.format(K), cov_factor_scale_init[-1,:], constraint=constraints.positive)
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init)
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    return loc, cov_factor, cov_diag

def zeroMeanFactor(X, batch_size, prior_parameters):
    """
    Parameters are K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale
    """
    N, D = X.shape
    K, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = prior_parameters[0]
    with pyro.plate('D', D):
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1):
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc[:K-1,:],cov_factor_scale[:K-1,:]))
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_loc[-1,:],cov_factor_scale[-1,:]))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
        loc = torch.zeros(D)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(loc, cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X

zeroMeanFactor.param_shapes_and_support = {'scaleloc':(('D',),constraints.real), 'scalescale':(('D',),constraints.positive), 'cov_factor_loc':(('K','D'),constraints.real), 'cov_factor_scale':(('K','D'),constraints.positive)}

def zeroMeanFactor2(X, batch_size, prior_parameters):
    """
    Parameters are K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale

    NEED TO CHECK ALL SHAPES FOR SUPERFLUOUS SINGLETON DIMENSIONS
    """
    N, D = X.shape
    K, scalelocinit, scalescaleinit, cov_factor_loc_init, cov_factor_scale_init = prior_parameters[0]
    with pyro.plate('D', D, dim=-1):
        #cov_diag_loc = pyro.param('scale_loc_prior', scalelocinit, constraint=constraints.positive)
        cov_diag_loc = pyro.param('scale_loc_prior', scalelocinit)
        cov_diag_scale = pyro.param('scale_scale_prior', scalescaleinit, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(cov_diag_loc, cov_diag_scale))
        cov_diag = cov_diag*torch.ones(D)
        # sample variables
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1, dim=-2):
                cov_factor_loc = pyro.param('cov_factor_prior_loc_{}'.format(K), cov_factor_loc_init[:K-1,:])
                cov_factor_scale = pyro.param('cov_factor_prior_scale_{}'.format(K), cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor_new_loc = pyro.param('cov_factor_new_loc_prior_{}'.format(K), cov_factor_loc_init[-1,:])
            cov_factor_new_scale = pyro.param('cov_factor_new_scale_prior_{}'.format(K), cov_factor_scale_init[-1,:], constraint=constraints.positive)
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
            # when using pyro.infer.Predictive, cov_factor_new is somehow sampled as 2-d tensors instead of 1-d
            #print(cov_factor.shape)
            #print(cov_factor_new.shape)
            if cov_factor_new.dim() == cov_factor.dim():
                cov_factor = torch.cat([cov_factor, cov_factor_new], dim=-2)
                #cov_factor = torch.cat([cov_factor, cov_factor_new], dim=1)
            else:
                cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=-2)], dim=-2)
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_prior_loc_{}'.format(K), cov_factor_loc_init)
                cov_factor_scale = pyro.param('cov_factor_prior_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(-2,-1)
    with pyro.plate('N', size=N, subsample_size=batch_size, dim=-1) as ind:
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(torch.zeros(D), cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X

def zeroMeanFactorGuide(X, batch_size, variational_parameter_initialization):
    N, D = X.shape
    K, scaleloc, scalescale, cov_factor_loc_init, cov_factor_scale_init = variational_parameter_initialization[1]
    with pyro.plate('D', D, dim=-1):
        cov_diag_loc = pyro.param('scale_loc', scaleloc, constraint=constraints.positive)
        cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(cov_diag_loc, cov_diag_scale))
        cov_diag = cov_diag*torch.ones(D)
        # sample variables
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1, dim=-2):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init[:K-1,:])
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init[:K-1,:], constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor_new_loc = pyro.param('cov_factor_new_loc_{}'.format(K), cov_factor_loc_init[-1,:])
            cov_factor_new_scale = pyro.param('cov_factor_new_scale_{}'.format(K), cov_factor_scale_init[-1,:], constraint=constraints.positive)
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
            # when using pyro.infer.Predictive, cov_factor_new is somehow sampled as 2-d tensors instead of 1-d
            if cov_factor_new.dim() == cov_factor.dim():
                cov_factor = torch.cat([cov_factor, cov_factor_new], dim=-2)
            else:
                cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=-2)], dim=-2)
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init)
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(-2,-1)
    return cov_factor, cov_diag

def factorLocalLatents(X, batch_size, prior_parameters):
    N, D = X.shape
    K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale = prior_parameters[0]
    cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
    cov_diag = torch.diag(cov_diag*torch.ones(D))
    with pyro.plate('D', D):
        loc = pyro.sample('loc', dist.Normal(locloc, locscale))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        #cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        with pyro.plate('K', K):
            local_latent = pyro.sample('local_latent', dist.Normal(0.,1.))
        wz = torch.matmul(local_latent.transpose(0,1), cov_factor)
        X = pyro.sample('obs', dist.MultivariateNormal(wz + loc, covariance_matrix=cov_diag), obs=X.index_select(0, ind))
    return X


def projectedMixture(X, batch_size, prior_parameters):
    """
    Covariances of all clusters are locked, we're just learning one covariance, mixture weights and means
    """
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale, component_logits_concentration, cov_factor_loc,cov_factor_scale = prior_parameters[0]
    C = locloc.shape[0]
    K = cov_factor_loc.shape[0]
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        with pyro.plate('K', K):
            cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
        with pyro.plate('C', C):
            locs = pyro.sample('locs', dist.Normal(locloc,locscale))
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        assignment = pyro.sample('assignment', dist.Categorical(component_logits), infer={"enumerate": "parallel"})
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(locs.index_select(-2, assignment), cov_factor, cov_diag), obs=X.index_select(0, ind))
    return X

def projectedMixtureGuide(X, batch_size, variational_parameter_initialization):
    """
    Covariances of all clusters are locked, we're just learning mixture weights and means
    """
    N, D = X.shape
    loc_loc, loc_scale, scale_loc, scale_scale, component_logits_concentration, cov_factor_loc_init,cov_factor_scale_init = variational_parameter_initialization[1]
    C = loc_loc.shape[0]
    K = cov_factor_loc_init.shape[0]
    component_logits_concentration = pyro.param('component_logits_concentration', component_logits_concentration, constraint=constraints.positive)
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag_loc = pyro.param('scale_loc', scale_loc)
        cov_diag_scale = pyro.param('scale_scale', scale_scale, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(scale_loc, scale_scale))
        cov_diag = cov_diag*torch.ones(D)
        with pyro.plate('K', K):
            cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init)
            cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
            cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
        with pyro.plate('C', C):
            loc_loc = pyro.param('loc_loc', loc_loc)
            loc_scale = pyro.param('loc_scale', loc_scale, constraint=constraints.positive)
            locs = pyro.sample('locs', dist.Normal(loc_loc,loc_scale))
    return component_logits, cov_diag, cov_factor, locs


def sphericalMixture(X, batch_size, prior_parameters):
    """
    Covariances of all clusters are diagonal, only params are mixture weights, means and shared variable variances
    ...so it's actually an ellipsoidal mixture, if you wanna be anal about it. Cunt.
    """
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale, component_logits_concentration = prior_parameters[0]
    # get number of clusters from first dimension of means
    C = locloc.shape[0]
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        with pyro.plate('C', C):
            locs = pyro.sample('locs', dist.Normal(locloc,locscale))
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        assignment = pyro.sample('assignment', dist.Categorical(component_logits), infer={"enumerate": "parallel"})
        #X = pyro.sample('obs', dist.MultivariateNormal(locs.index_select(-2, assignment), torch.diag(cov_diag)), obs=X.index_select(0, ind))
        # use index_select instead of locs[assignment] so batching works correctly
        # have to select the right index both in a parallel enumeration context and a batch context
        # leading to this ugly hack
        indexed_locs = locs.index_select(-2, assignment.squeeze()).view(*locs.shape[:-2],*assignment.shape,locs.shape[-1])
        #print(locs.shape)
        #print(assignment.shape)
        #print(indexed_locs.shape)
        X = pyro.sample('obs', dist.MultivariateNormal(indexed_locs, torch.diag_embed(cov_diag)), obs=X.index_select(0, ind))
    return X

def sphericalMixtureGuide(X, batch_size, variational_parameter_initialization):
    """
    Covariances of all clusters are diagonal, only params are mixture weights, means and a shared variable variances
    """
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale, component_logits_concentration = variational_parameter_initialization[1]
    C = locloc.shape[0]
    component_logits_concentration = pyro.param('component_logits_concentration', component_logits_concentration, constraint=constraints.positive)
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag_loc = pyro.param('scale_loc', scaleloc)
        cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        with pyro.plate('C', C):
            locloc = pyro.param('loc_loc', locloc)
            locscale = pyro.param('loc_scale', locscale, constraint=constraints.positive)
            locs = pyro.sample('locs', dist.Normal(locloc,locscale))
    return component_logits, cov_diag, locs

def scaledSphericalMixture(X, batch_size, prior_parameters):
    """
    Covariances of all clusters are diagonal, only params are mixture weights, means and shared variable variances
    ...so it's actually an ellipsoidal mixture, if you wanna be anal about it. Cunt.
    """
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale, covscalingloc, covscalingscale, component_logits_concentration = prior_parameters[0]
    # get number of clusters from first dimension of means
    C = locloc.shape[0]
    Cplate = pyro.plate('C', C)
    component_logits_concentration = pyro.param('component_logits_concentration_prior', component_logits_concentration, constraint=constraints.positive)
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag_loc = pyro.param('scale_loc_prior', scaleloc)
        cov_diag_scale = pyro.param('scale_scale_prior', scalescale, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        with Cplate:
            locloc = pyro.param('loc_loc_prior', locloc)
            locscale = pyro.param('loc_scale_prior', locscale, constraint=constraints.positive)
            locs = pyro.sample('locs', dist.Normal(locloc,locscale))
    with Cplate:
        cov_scaling_loc = pyro.param('cov_scaling_loc_prior', covscalingloc)
        cov_scaling_scale = pyro.param('cov_scaling_scale_prior', covscalingscale, constraint=constraints.positive)
        cov_scaling = pyro.sample('cov_scaling', dist.LogNormal(covscalingloc, covscalingscale))
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        assignment = pyro.sample('assignment', dist.Categorical(component_logits), infer={"enumerate": "parallel"})
        #X = pyro.sample('obs', dist.MultivariateNormal(locs.index_select(-2, assignment), torch.diag(cov_diag)), obs=X.index_select(0, ind))
        # use index_select instead of locs[assignment] so batching works correctly
        # have to select the right index both in a parallel enumeration context and a batch context
        # leading to this ugly hack
        indexed_locs = locs.index_select(-2, assignment.squeeze()).view(*locs.shape[:-2],*assignment.shape,locs.shape[-1])
        #print(locs.shape)
        #print(assignment.shape)
        #print(indexed_locs.shape)
        scaled_covariances = torch.diag_embed(cov_scaling[assignment]*cov_diag)
        X = pyro.sample('obs', dist.MultivariateNormal(indexed_locs, scaled_covariances), obs=X.index_select(0, ind))
    return X

def scaledSphericalMixtureGuide(X, batch_size, variational_parameter_initialization):
    """
    Covariances of all clusters are diagonal, only params are mixture weights, means and shared variable variances
    ...so it's actually an ellipsoidal mixture, if you wanna be anal about it. Cunt.
    """
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale, covscalingloc, covscalingscale, component_logits_concentration = variational_parameter_initialization[1]
    # get number of clusters from first dimension of means
    C = locloc.shape[0]
    Cplate = pyro.plate('C', C)
    component_logits_concentration = pyro.param('component_logits_concentration', component_logits_concentration, constraint=constraints.positive)
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag_loc = pyro.param('scale_loc', scaleloc)
        cov_diag_scale = pyro.param('scale_scale', scalescale, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        with Cplate:
            locloc = pyro.param('loc_loc', locloc)
            locscale = pyro.param('loc_scale', locscale, constraint=constraints.positive)
            locs = pyro.sample('locs', dist.Normal(locloc,locscale))
    with Cplate:
        cov_scaling_loc = pyro.param('cov_scaling_loc', covscalingloc)
        cov_scaling_scale = pyro.param('cov_scaling_scale', covscalingscale, constraint=constraints.positive)
        cov_scaling = pyro.sample('cov_scaling', dist.LogNormal(covscalingloc, covscalingscale))
    return component_logits, cov_diag, cov_scaling, locs