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
D = 2
K = 1
N = 100

# dataset settings
#D = 10
#K = 4
#N = 5000
seed = 45

D = 110
K = 80
N = 5000

#D = 500
#K = 160
#N = 10000

# generate data if it hasn't been already
dataset_filename =generate_dataset(D, N, K, seed)

# define experiment
if D < 20:
    Ks = range(1,D+1)
else:
    Ks = range(1,D,10)
#restarts = range(10)
restarts = range(10,20)
inits = ['rng','pca','inc','ard']

all_jobs = []

# make list of all jobs to be completed
for K in Ks:
    for restart in restarts:
        for init in inits:
            # the first K in incremental initialization is the same as random initialization
            if init == 'inc' and K == 1:
                continue
            current_K_index = Ks.index(K)
            previous_K = Ks[current_K_index-1]
            all_jobs.append((K, previous_K, restart, init))

# this loop ranks remaining jobs based on the state of the folder
# by priority and trains the next one
# we start with all jobs that need to be done, in case some jobs were started but failed
jobs = all_jobs

while jobs:
    jobs = all_jobs
    # prune those jobs that have been completed, or are being worked on
    pruned_jobs = jobs.copy()
    for job in jobs:
        K, previous_K, restart, init = job
        if init == 'ard':
            _id = '_'.join([str(restart), init])
        else:
            _id = '_'.join([str(K), str(restart), init])
        if os.path.exists(_id + '.p') or os.path.exists(_id + 'started'):
            pruned_jobs.remove(job)
    jobs = pruned_jobs
    # if there are jobs left rank them
    if not jobs:
        print("All jobs are completed.")
        sys.exit()

    # rank first by seed, then by K
    jobs = sorted(jobs, key=lambda tup: (tup[2],tup[0]))
    
    # highest priority to rng K=1 for all seeds
    for i,job in enumerate(jobs):
        jobs[i] = job + ((job[-1] == 'rng' and job[0] == 1),)
    jobs = sorted(jobs, key=lambda tup: tup[-1], reverse=True)

    top_priority = 0
    # train the top priority job
    # unless it's an inc job and its teacher doesn't exist yet
    inc_and_teacher_not_present = True
    while inc_and_teacher_not_present:
        top_priority_job = jobs[top_priority][:-1]
        if top_priority_job[-1] == 'inc':
            K, previous_K, restart, init = top_priority_job
            teacher_id = '_'.join([str(previous_K), str(restart), init])
            teacher_present = os.path.exists(teacher_id + '.p')
            if not teacher_present:
                top_priority += 1
            else:
                inc_and_teacher_not_present = False
        else:
            inc_and_teacher_not_present = False
    try:
        print(f'Training {top_priority_job[-1]} init with {top_priority_job[0]} factors for seed {top_priority_job[2]}.')
        train_job(dataset_filename, *top_priority_job)
    except Exception as e:
        # mark failure of training job
        fail_filename = _id + 'failed'
        with open(fail_filename, 'w') as f:
            f.write(str(e))