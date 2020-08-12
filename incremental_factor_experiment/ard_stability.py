import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
import torch
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from pyro import distributions as dist
from pyro.infer.predictive import Predictive, _guess_max_plate_nesting
import sys
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide.initialization import init_to_sample
import operator
sys.path.append("..")
sys.path.append("../ppca")
pyro.enable_validation(True) 

from tracepredictive import *
from inference import *
from initializations import *
from models_and_guides import *
from plotting import *
from utils import *

# set seed to generate the same dataset every time
seed = 45
pyro.set_rng_seed(seed)

D = 10
N = 5000
K = 4

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

zeroMeanFactor_teacher = ZeroMeanFactor(torch.empty(N,D), K, N, '0')

dgp = zeroMeanFactor_teacher.unconditioned_model
trace = pyro.poutine.trace(dgp).get_trace(torch.empty(N,D))
true_variables = dict([(name,trace.nodes[name]["value"]) for name in trace.stochastic_nodes if len(name)>1])
X = true_variables['obs'].detach()

data, test_data = train_test_split(X)

all_inference_results = []

seed = 0
for seed in range(1,10):
    pyro.set_rng_seed(seed)

    ARD = ZeroMeanFactorARD(data, 16, '3')

    inference_args = (ARD, data, test_data, config)

    inference_results = inference(*inference_args)

    all_inference_results.append((ARD,inference_results))

filename = 'ard_stability.p'
with open(filename, 'wb') as f:
	pickle.dump([], f)