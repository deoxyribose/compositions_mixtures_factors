from __future__ import absolute_import, division, print_function

import numbers
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

from six import add_metaclass

import torch
import pyro.poutine as poutine
from pyro.distributions import Categorical, Empirical
from pyro.ops.stats import waic
from pyro.infer.util import site_is_subsample
from pyro.infer import *

class TracePredictive(TracePosterior):
    """
    Generates and holds traces from the posterior predictive distribution,
    given model execution traces from the approximate posterior. This is
    achieved by constraining latent sites to randomly sampled parameter
    values from the model execution traces and running the model forward
    to generate traces with new response ("_RETURN") sites.
    :param model: arbitrary Python callable containing Pyro primitives.
    :param TracePosterior posterior: trace posterior instance holding samples from the model's approximate posterior.
    :param int num_samples: number of samples to generate.
    :param keep_sites: The sites which should be sampled from posterior distribution (default: all)
    """
    def __init__(self, model, posterior, num_samples, keep_sites=None):
        self.model = model
        self.posterior = posterior
        self.num_samples = num_samples
        self.keep_sites = keep_sites
        super(TracePredictive, self).__init__()

    def _traces(self, *args, **kwargs):
        if not self.posterior.exec_traces:
            self.posterior.run(*args, **kwargs)
        data_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        for _ in range(self.num_samples):
            model_trace = self.posterior().copy()
            self._remove_dropped_nodes(model_trace)
            #self._adjust_to_data(model_trace, data_trace)
            resampled_trace = poutine.trace(poutine.replay(self.model, model_trace)).get_trace(*args, **kwargs)
            yield (resampled_trace, 0., 0)

    def _remove_dropped_nodes(self, trace):
        if self.keep_sites is None:
            return
        for name, site in list(trace.nodes.items()):
            if name not in self.keep_sites:
                trace.remove_node(name)
                continue

    def _adjust_to_data(self, trace, data_trace):
        subsampled_idxs = dict()
        for name, site in trace.iter_stochastic_nodes():
            print(site["name"],site["value"])
            # Adjust subsample sites
            if site_is_subsample(site):
                site["fn"] = data_trace.nodes[name]["fn"]
                site["value"] = data_trace.nodes[name]["value"]
            else:
                # Adjust sites under conditionally independent stacks
                orig_cis_stack = site["cond_indep_stack"]
                site["cond_indep_stack"] = data_trace.nodes[name]["cond_indep_stack"]
                assert len(orig_cis_stack) == len(site["cond_indep_stack"])
                site["fn"] = data_trace.nodes[name]["fn"]
                for ocis, cis in zip(orig_cis_stack, site["cond_indep_stack"]):
                    # Select random sub-indices to replay values under conditionally independent stacks.
                    # Otherwise, we assume there is an dependence of indexes between training data
                    # and prediction data.
                    assert ocis.name == cis.name
                    assert not site_is_subsample(site)
                    batch_dim = cis.dim - site["fn"].event_dim
                    subsampled_idxs[cis.name] = subsampled_idxs.get(cis.name,
                                                                    torch.randint(0, ocis.size, (cis.size,),
                                                                                  device=site["value"].device))
                    site["value"] = site["value"].index_select(batch_dim, subsampled_idxs[cis.name])
            print(subsampled_idxs)
            print(site["name"],site["value"])