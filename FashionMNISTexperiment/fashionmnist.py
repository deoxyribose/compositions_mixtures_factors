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
import argparse
sys.path.append("..")
sys.path.append("../factor/")
from tracepredictive import *
from inference import *
from models_and_guides import *
from large_joint_optim import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for incremental inference in ppca')
    parser.add_argument('initseed', type=int, help='Random seed to set for each K')
    parser.add_argument('K_range', type=int, nargs=2, help='', default=[1,2])
    args = parser.parse_args()

    Kmin, Kmax = args.K_range

    assert Kmin < Kmax

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

    n_experimental_conditions = 2
    experimental_condition = 0
    max_n_iter = 1000
    n_posterior_samples = 1600
    # optimization parameters
    n_multistart = 1
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


    for experimental_condition in range(n_experimental_conditions):
        # fashionmnist.py detects pickle files in the folder it's run in
        # if e.g. 1.p, 2.p and 3.p all exist, it will run the K=4 model and create 4.p
        for K in range(Kmin, Kmax+1):
            pyro.set_rng_seed(args.initseed)
            # detect K.p file
            if os.path.exists('{}.p'.format(K)) and experimental_condition == 0:
                continue
            # if K.p doesn't exist, create it
            with open('{}.p'.format(K), 'wb') as f:
                pickle.dump(([None]), f)
            # if pickle file exists, likewise continue
            filename = "{}_factors_{}_fashionMNIST.p".format(K,str(experimental_condition))
            if experimental_condition > 0 and K == Kmin:
                Kminmodel = "{}_factors_{}_fashionMNIST.p".format(Kmin,0)
                restarts, results = pickle.load(open(Kminmodel, 'rb'))
                best_lppd_at_convergence = np.inf
                for result in results:
                    _,_,lppds,param_history,_,_ = result
                    mean_lppd_at_convergence = sum(lppds[-10:])/10
                    if mean_lppd_at_convergence < best_lppd_at_convergence:
                        best_lppd_at_convergence = mean_lppd_at_convergence
                        best_param_history = param_history
                param_history = best_param_history
                continue
            elif experimental_condition == 0:
                param_history = None
            if os.path.exists(filename):
                print('{} exists, loading and continueing'.format(filename))
                continue
            for restart in range(n_multistart):
                # if some but not all restarts are completed
                # load the finished restarts, and train the next one
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
                init = get_h_and_v_params(K, D, experimental_condition, 1, data, param_history)
                # run 300 iterations
                inference_result = inference(zeroMeanFactor2, zeroMeanFactorGuide, data, test_data, init, max_n_iter, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                inference_results.append(inference_result)
                with open(filename, 'wb') as f:
                    pickle.dump((restart,inference_results), f)
            os.remove('{}.p'.format(K))