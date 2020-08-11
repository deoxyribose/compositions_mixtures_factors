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
import re

# define inference config
config = dict(
    n_iter = 700,
    learning_rate = 0.1,
    beta1 = 0.9,
    beta2 = 0.999,
    learning_rate_decay = 0.9999,
    batch_size = 16,
    n_elbo_particles = 16,
    n_posterior_samples = 512,
    window = 10,
    convergence_window = 15,
    slope_significance = 1,#0.1,
    track_params = True,
    monitor_gradients = True,
    telemetry = None,
)

def get_best_teacher(K):
	# get list of all filenames
	mypath = './'
	if K == 2:
		teacher_init = 'rng'
	else:
		teacher_init = 'inc'
	regex = re.compile(str(K-1)+'_\d_'+teacher_init+'.p')
	previous_K_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and re.match(regex, f)]
	
	models = []
	average_MNLL_at_convergences = []
	# iterate through all restarts of previous model
	for f in previous_K_files:
	    with open(f, 'rb') as f:
	        model, telemetry = pickle.load(f)
			# find average MNLL in convergence window
	        average_MNLL_at_convergence = sum(telemetry['MNLL'][-config['convergence_window']:])/config['convergence_window']
	        average_MNLL_at_convergences.append(average_MNLL_at_convergence)
	        models.append(model)
	# return model with lowest average MNLL at convergence
	return models[np.array(average_MNLL_at_convergences).argmin()]

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

def train_job(dataset_filename, K, restart, init):

	# load data
	with open(dataset_filename, 'rb') as f:
		data, test_data, true_variables = pickle.load(f)


	# FOR TESTING
	#config['n_iter'] = 16
	#config['n_posterior_samples'] = 16

	# set id, initialize parameters
	_id = '_'.join([str(K), str(restart), init])
	model = ZeroMeanFactor(data, K, config['batch_size'], _id)
	if init == 'pca':
		pca_init(model, data)
	elif init == 'inc':
		if K > 1:
			teacher = get_best_teacher(K)
			incremental_init(model, teacher)
	elif init == 'ard':
		_id = '_'.join([str(restart), init])
		model = ZeroMeanFactorARD(data, config['batch_size'], _id)
	elif init == 'rng':
		pass
	else:
		print("Invalid init string.")

	# mark start of training job
	start_filename = _id + 'started'
	with open(start_filename, 'wb') as f:
		pickle.dump([], f)


	inference_args = (model, data, test_data, config)
	inference_results = inference(*inference_args)

	filename = _id + '.p'
	with open(filename, 'wb') as f:
		pickle.dump((model, inference_results), f)

	if os.path.exists(start_filename):
		os.remove(start_filename)