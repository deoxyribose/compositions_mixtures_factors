import argparse
import numpy as np
import torch
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from pyro import distributions as dist
from collections import defaultdict
import matplotlib.pylab as plt
import time
import pickle
import os
import pandas as pd
import xarray as xr
import scipy.stats as sps
import argparse
import gc
import sys
sys.path.append("..")
from tracepredictive import *
from inference import *
from models_and_guides import *
from initializations import *
from utils import *

def generate_dataset(D, N, K, seed):
	filename = "incremental_factor_experiment_dataset" + '_'.join(map(str, [D,N,K,seed])) + '.p'

	if not os.path.exists(filename):
		# set seed to generate the same dataset every time
		pyro.set_rng_seed(seed)

		zeroMeanFactor_teacher = ZeroMeanFactor(torch.empty(N,D), K, N, '0')

		dgp = zeroMeanFactor_teacher.unconditioned_model
		trace = pyro.poutine.trace(dgp).get_trace(torch.empty(N,D))
		true_variables = dict([(name,trace.nodes[name]["value"]) for name in trace.stochastic_nodes if len(name)>1])
		X = true_variables['obs'].detach()

		data, test_data = train_test_split(X)

		with open(filename, 'wb') as f:
			pickle.dump((data, test_data, true_variables), f)

	return filename