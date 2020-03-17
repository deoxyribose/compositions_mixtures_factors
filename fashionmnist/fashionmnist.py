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
import time
sys.path.append("..")
sys.path.append("../factor/")
from tracepredictive import *
from inference import *
from models_and_guides import *
from large_joint_optim import *

def get_param_history_of_best_restart(model):
    print("Loading best restart from {}".format(model))
    with open(model, 'rb') as f:
        results = pickle.load(f)
    best_lppd_at_convergence = np.inf
    for result in results:
        _,_,lppds,param_history,_,_ = result
        mean_lppd_at_convergence = sum(lppds[-10:])/10
        if mean_lppd_at_convergence < best_lppd_at_convergence:
            best_lppd_at_convergence = mean_lppd_at_convergence
            best_param_history = param_history
    return best_param_history

def sleep_until_file_exists(file):
    seconds_passed = 0
    while not os.path.exists(file) or seconds_passed > 3600*2:
        print("Going to sleep until {} exists".format(file))
        time.sleep(1)
        seconds_passed += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments for incremental inference in ppca')
    parser.add_argument('initseed', type=int, help='Random seed to set for each K')
    parser.add_argument('K_range', type=int, nargs=2, help='', default=[1,2])
    parser.add_argument('smoke_test', type=bool, nargs='?', const = True, default=False)
    args = parser.parse_args()

    Kmin, Kmax = args.K_range

    smoke_test = args.smoke_test

    assert Kmin < Kmax

    dl = DataLoader(torchvision.datasets.FashionMNIST('../dataforfashionmnist', download=True, train=True))
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

    n_conditions = 2
    learning_rate = 0.05
    momentum1 = 0.9
    momentum2 = 0.999
    decay = 1.
    batch_size = 10
    n_mc_samples = 16
    slope_significance = 1. # p_value of slope has to be smaller than this for training to continue

    if smoke_test:
        print("Running smoke test")
        max_n_iter = 30
        n_posterior_samples = 200
        n_multistart = 2
        window = 10 # compute lppd every window iterations
        convergence_window = 5 # estimate slope of convergence_window lppds
        Kmin = 1
        Kmax = 3
    else:
        max_n_iter = 1000
        n_posterior_samples = 1000
        # optimization parameters
        n_multistart = 5
        window = 10 # compute lppd every window iterations
        convergence_window = 30 # estimate slope of convergence_window lppds

    initseed = 42
    K = 1
    D = x_train.shape[1]
    data = x_train
    test_data = x_valid

    for condition in range(n_conditions):
        print("Condition {}".format(condition))
        for K in range(Kmin, Kmax+1):
            pyro.set_rng_seed(args.initseed)

            # load the previous model
            if condition == 1:
                if K == Kmin:
                    Kminmodel = "{}_factors_{}_fashionMNIST.p".format(Kmin,0)
                    sleep_until_file_exists(Kminmodel)
                    param_history = get_param_history_of_best_restart(Kminmodel)
                    continue
                elif K > Kmin+1:
                    prevmodel = "{}_factors_{}_fashionMNIST.p".format(K-1,1)
                    sleep_until_file_exists(prevmodel)
                    param_history = get_param_history_of_best_restart(prevmodel)
                    print(prevmodel)
            elif condition == 0:
                param_history = None

            # identify an uninitiated K
            for restart in range(1,n_multistart+1):
                if os.path.exists('Condition{}_K{}_Restart{}.p'.format(condition, K, restart)):
                    print('Restart {} for K {} in condition {} begun, continuing to next restart'.format(restart, K, condition))
                    continue

                print('Multistart {}/{}'.format(restart,n_multistart))
                # mark that a server has begun this restart
                with open('Condition{}_K{}_Restart{}.p'.format(condition, K, restart), 'wb') as f:
                    pickle.dump(([None]), f)
                # initialize
                init = get_h_and_v_params(K, D, condition, 1, data, param_history)
                #if smoke_test:
                    #print('Init is {}'.format(init[0][1]    ))
                inference_result = inference(zeroMeanFactor2, zeroMeanFactorGuide, data, test_data, init, max_n_iter, window, batch_size, n_mc_samples, learning_rate, decay, n_posterior_samples, slope_significance)
                print(inference_result[4][1][:10])
                # save the restart
                restart_filename = "{}_restart_{}_factors_{}_fashionMNIST.p".format(restart,K,condition)
                with open(restart_filename, 'wb') as f:
                    print("Saving restart {} to restart file".format(restart))
                    pickle.dump((restart,inference_result), f)

            # aggregate restarts
            if all([os.path.exists("{}_restart_{}_factors_{}_fashionMNIST.p".format(restart,K,condition)) for restart in range(1,n_multistart+1)]):
                inference_results = []
                # aggregate results
                for restart in range(1,n_multistart+1):
                    restart_filename = "{}_restart_{}_factors_{}_fashionMNIST.p".format(restart,K,condition)
                    with open(restart_filename, 'rb') as f:
                        restarts, results = pickle.load(f)
                    inference_results.append(results)
                # save as one pickle
                filename = "{}_factors_{}_fashionMNIST.p".format(K,condition)
                print("Aggregating restarts in {}".format(filename))
                with open(filename, 'wb') as f:
                    pickle.dump(inference_results, f)
    # clean-up
    for condition in range(n_conditions):
        for restart in range(1,n_multistart+1):
            for K in range(Kmin, Kmax+1):
                scheduling_filename = 'Condition{}_K{}_Restart{}.p'.format(condition, K, restart)
                restart_filename = "{}_restart_{}_factors_{}_fashionMNIST.p".format(restart,K,condition)
                filenames = [scheduling_filename, restart_filename]
                for filename in filenames:
                    if os.path.exists(filename):
                        os.remove(filename)