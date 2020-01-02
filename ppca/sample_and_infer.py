import numpy as np
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pylab as plt
from IPython.display import display, clear_output
import torch
import torchvision
import scipy.stats as sps
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from pyro import distributions as dst
from collections import defaultdict
import sys
sys.path.append("..")

from tracepredictive import *
from inference import *
from models_and_guides import *
from plotting import *

pyro.set_rng_seed(42)
N, D = 1000, 2
K = 2
locloc = torch.zeros(1,D)
locscale = torch.ones(1,D)*5
scaleloc = torch.zeros(D)
scalescale = torch.ones(D)*0.1
cov_factor_loc = torch.zeros(K,D)
cov_factor_scale = torch.ones(K,D)
tmp = (K, locloc, locscale, scaleloc, scalescale, cov_factor_loc,cov_factor_scale)
init = (tmp, tmp)
unconditioned = pyro.poutine.uncondition(factor)
trace = pyro.poutine.trace(unconditioned).get_trace(torch.zeros(N,D), N, init)
true_variables = [trace.nodes[name]["value"] for name in trace.stochastic_nodes]

data = true_variables[-1]

lr = 0.3
lrd = 1.
max_n_iter = 20000
x_train = data[:800]
x_valid = data[800:]

inference_results = []
for seed in range(100,105):
    # svi, losses, lppds, param_history, init, gradient_norms
    inference_result = inference(factor, factorGuide, x_train, x_valid, init, learning_rate= lr, learning_rate_decay = lrd, n_iter = max_n_iter, window=10)
    inference_results.append(inference_result)

for inference_result in inference_results:
	svi, losses, lppds, param_history, init, gradient_norms = inference_result
	print(lppds[-1])