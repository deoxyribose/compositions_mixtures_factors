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
import sys
sys.path.append("..")

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
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(torch.zeros(D), cov_factor=cov_factor, cov_diag=cov_diag), obs=X.index_select(0, ind))
    return X


def zeroMeanFactor2(X, batch_size, prior_parameters):
    """
    Parameters are K, locloc, locscale, scaleloc, scalescale, cov_factor_loc, cov_factor_scale
    """
    N, D = X.shape
    K, scalelocinit, scalescaleinit, cov_factor_locinit, cov_factor_scaleinit = prior_parameters[0]
    with pyro.plate('D', D, dim=-1):
        cov_diag_loc = pyro.param('scale_loc_hyper', scalelocinit, constraint=constraints.positive)
        cov_diag_scale = pyro.param('scale_scale_hyper', scalescaleinit, constraint=constraints.positive)
        cov_diag = pyro.sample('scale', dist.LogNormal(cov_diag_loc, cov_diag_scale))
        cov_diag = cov_diag*torch.ones(D)
        # sample variables
        cov_factor = None
        if K > 1:
            with pyro.plate('K', K-1, dim=-2):
                cov_factor_loc = pyro.param('cov_factor_loc_hyper_{}'.format(K), cov_factor_locinit[:K-1,:])
                cov_factor_scale = pyro.param('cov_factor_scale_hyper_{}'.format(K), cov_factor_scaleinit[:K-1,:], constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc, cov_factor_scale))
            cov_factor_new_loc = pyro.param('cov_factor_new_loc_hyper_{}'.format(K), cov_factor_locinit[-1,:])
            cov_factor_new_scale = pyro.param('cov_factor_new_scale_hyper_{}'.format(K), cov_factor_scaleinit[-1,:], constraint=constraints.positive)
            cov_factor_new = pyro.sample('cov_factor_new', dist.Normal(cov_factor_new_loc,cov_factor_new_scale))
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_loc_hyper_{}'.format(K), cov_factor_locinit)
                cov_factor_scale = pyro.param('cov_factor_scale_hyper_{}'.format(K), cov_factor_scaleinit, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
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
            cov_factor = torch.cat([cov_factor, torch.unsqueeze(cov_factor_new, dim=0)])
        else:
            with pyro.plate('K', K):
                cov_factor_loc = pyro.param('cov_factor_loc_{}'.format(K), cov_factor_loc_init)
                cov_factor_scale = pyro.param('cov_factor_scale_{}'.format(K), cov_factor_scale_init, constraint=constraints.positive)
                cov_factor = pyro.sample('cov_factor', dist.Normal(cov_factor_loc,cov_factor_scale))
        cov_factor = cov_factor.transpose(0,1)
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
    Covariances of all clusters are locked, we're just learning mixture weights and means
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
        X = pyro.sample('obs', dist.LowRankMultivariateNormal(locs[assignment], cov_factor, cov_diag), obs=X.index_select(0, ind))
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
    Covariances of all clusters are diagonal, we're just learning mixture weights and means
    """
    N, D = X.shape
    locloc, locscale, scaleloc, scalescale, component_logits_concentration = prior_parameters[0]
    C = locloc.shape[0]
    component_logits = pyro.sample('component_logits', dist.Dirichlet(component_logits_concentration))
    with pyro.plate('D', D):
        cov_diag = pyro.sample('scale', dist.LogNormal(scaleloc, scalescale))
        with pyro.plate('C', C):
            locs = pyro.sample('locs', dist.Normal(locloc,locscale))
    with pyro.plate('N', size=N, subsample_size=batch_size) as ind:
        assignment = pyro.sample('assignment', dist.Categorical(component_logits), infer={"enumerate": "parallel"})
        X = pyro.sample('obs', dist.MultivariateNormal(locs[assignment], torch.diag(cov_diag)), obs=X.index_select(0, ind))
    return X

def sphericalMixtureGuide(X, batch_size, variational_parameter_initialization):
    """
    Covariances of all clusters are diagonal, we're just learning mixture weights and means
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