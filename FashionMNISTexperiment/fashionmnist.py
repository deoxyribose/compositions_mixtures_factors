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
sys.path.append("../factor/")
from tracepredictive import *
from inference import *
from models_and_guides import *
from large_joint_optim import *

dl = DataLoader(torchvision.datasets.FashionMNIST('../fashionmnist', download=True, train=True))
tensor = dl.dataset.train_data
tensor = tensor.to(dtype=torch.float32)
tr = tensor.reshape(tensor.size(0), -1) 
tr = tr/128
targets = dl.dataset.train_labels
targets = targets.to(dtype=torch.long)

x_train = tr[0:50000]
y_train = targets[0:50000]
x_valid = tr[50000:60000]
y_valid = targets[50000:60000]

idx = np.random.choice(x_train.shape[0])
plt.imshow(x_train[idx,:].reshape(-1,28));

n_experimental_conditions = 2
experimental_condition = 0
max_n_iter = 1000
n_posterior_samples = 1600
# optimization parameters
n_multistart = 10
learning_rate = 0.05
momentum1 = 0.9
momentum2 = 0.999
decay = 1.
batch_size = 10
n_mc_samples = 16
window = 10 # compute lppd every window iterations
convergence_window = 10 # estimate slope of convergence_window lppds
slope_significance = 1. # p_value of slope has to be smaller than this for training to continue

initseed = 42
K = 1
D = x_train.shape[1]
data = x_train
test_data = x_valid

init = get_h_and_v_params(K,D,0,1,data)

for K in range(1,25):
    if os.path.exists('{}.p'.format(K)):
        continue
    with open('{}.p'.format(K), 'wb') as f:
        pickle.dump(([None]), f)
    filename = "{}_factors_{}_fashionMNIST.p".format(K,str(experimental_condition))
    if os.path.exists(filename):
        print('{} exists, loading and continueing'.format(filename))
        continue
    for restart in range(n_multistart):
        if os.path.exists(filename):
            print('{} completed, loading and continueing'.format(restart+1))
            with open(filename, 'rb') as f:
                finished_restart,inference_results = pickle.load(f)
            if finished_restart >= restart:
                continue
        else:
            inference_results = []
        print('Multistart {}/{}'.format(restart+1,n_multistart))
        pyro.clear_param_store()
        # initialize
        init = get_h_and_v_params(K, D, 0, 1, data)
        # run 300 iterations
        inference_result = inference(zeroMeanFactor2, zeroMeanFactorGuide, data, test_data, init, max_n_iter, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
        inference_results.append(inference_result)
        with open(filename, 'wb') as f:
            pickle.dump((restart,inference_results), f)
    os.remove('{}.p'.format(K))