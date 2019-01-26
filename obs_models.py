import torch
from torch.distributions import constraints
from collections import defaultdict


import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.optim import Adam
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO

from matplotlib import pyplot

import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)
pyro.enable_validation(True)

def independentGaussian(data):
    N, D = data.shape
    with pyro.plate('features', D):
        locs = pyro.sample('locs', pyro.distributions.Normal(0.,10.))
        scales = pyro.sample('scales', pyro.distributions.LogNormal(0.,4.))
        with pyro.plate('data', N):
            data = pyro.sample('obs', pyro.distributions.Normal(locs,scales), obs=data)
    return data

def independentCategorical(data):
    N, D = data.shape
    # compute number of categories in each feature
    C = [np.unique(data[:,i]).shape[0] for i in range(D)]
    Cmax = np.max(C)
    C = torch.tensor(C)
    # Since there can be different number of categories,
    # we let all features have the same categories as the most category-rich feature,
    # but set the prior of non-existent categories to epsilon.
    dirichlet_params = torch.tensor([np.r_[np.ones(c)*10*Cmax, np.ones(Cmax - c)*0.1*(1/Cmax)] for c in C])
    #for feature in pyro.plate('features', D):
    with pyro.plate('feature_plate', D):
        probs = pyro.sample('probs', pyro.distributions.Dirichlet(dirichlet_params))
        #probs = pyro.sample('probs_{}'.format(feature), pyro.distributions.Dirichlet(dirichlet_params))
        with pyro.plate('data_plate', N):
            data = pyro.sample('obs', pyro.distributions.Categorical(probs=probs),obs=data)
    return data

