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


from generate_dataset import generate_dataset
#from scheduler import run_next_job
from train_job import *

# test
#D = 2
#K = 1
#N = 100

# dataset settings
D = 10
K = 4
N = 5000
seed = 45

#D = 500
#K = 160
#N = 10000

# generate data if it hasn't been already
dataset_filename =generate_dataset(D, N, K, seed)

# define experiment
Ks = range(1,D+1)
restarts = range(10)
inits = ['rng','pca','inc','ard']

for K in Ks:
	for restart in restarts:
		for init in inits:
			_id = '_'.join([str(K), str(restart), init])
			# if the model is already trained or is in progress
			if os.path.exists(_id + '.p') or os.path.exists(_id + 'started'):
				continue
			# the first K in incremental initialization is the same as random initialization
			if init == 'inc' and K == 1:
				continue
			try:
				train_job(dataset_filename, K, restart, init)
			except RuntimeError:
				# mark failure of training job
				fail_filename = _id + 'failed'
				with open(fail_filename, 'wb') as f:
					pickle.dump([], f)