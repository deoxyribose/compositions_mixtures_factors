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
from tracepredictive import *
from models_and_guides import *

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
sys.path.append("..")

def plot_learning_curve(losses):
	plt.plot(losses,'b');

def plot_held_out_predictive(lppds):
	plt.plot(lppds);

def plot_parameter_evolution(param, param_history):
	D = param_history[param].shape[-1]
	n_iter = param_history[param].shape[0]
	idxs = np.random.choice(D, 10)
	plt.plot(np.squeeze(param_history[param].detach().numpy()[...,idxs]).reshape(n_iter, -1));
	plt.figure()

def plot_samples_from_model(model, svi, x_train, init):
	with torch.no_grad():
	    unconditioned_model = pyro.poutine.uncondition(model)
	    trace_pred = TracePredictive(unconditioned_model, svi, num_samples=1).run(x_train, 10, init)
	    predictive_dst_sample = [torch.unsqueeze(trace.nodes['obs']['value'],dim=0) for trace in trace_pred.exec_traces]
	for i in range(1):
	    for j in range(3):
	        plt.figure()
	        plt.imshow(predictive_dst_sample[i][0][j].reshape(-1,28).detach().numpy())

def plot_extremes_on_latent_dimensions(param_history, K, x_train, init):
	unconditioned_model = pyro.poutine.uncondition(factorLocalLatents)
	locmode = param_history['loc_loc'][-1].detach()
	cov_factor_loc = param_history['cov_factor_loc_{}'.format(K)][-1].detach()
	cov_factor_new_loc = param_history['cov_factor_new_loc_{}'.format(K)][-1].detach()
	cov_factormode = torch.cat([cov_factor_loc,torch.unsqueeze(cov_factor_new_loc,dim=0)])
	for i in range(K):
	    factor_coords = torch.ones(K)
	    factor_coords[i] = -1
	    zs = torch.matmul(torch.diag(100*factor_coords),torch.ones((K,10)))
	    condition_latent = pyro.poutine.condition(unconditioned_model, data={"loc":locmode,"cov_factor": cov_factormode,"local_latent": zs})
	    plt.imshow(condition_latent(x_train, 10, init)[0].reshape(-1,28).detach().numpy())
	    plt.figure()

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    mean_x, mean_y = mean
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)