import networkx as nx
import torch
import pyro
import pyro.distributions as dist
from model_operators import *
from code_generation import *

def marginalize_factor(factor_DAG):
    marg_factor = factor_DAG.copy()
    
    marg_factor.nodes['X']['distribution'] = dist.LowRankMultivariateNormal
    marg_factor.add_nodes_from([('loc',{'type':'function', 'function':torch.zeros, 'args':(Name(id='D'),)}),])
    marg_factor.add_nodes_from([('cov_factor_T',{'type':'function', 'function':torch.transpose, 'args':[Num(0),Num(1)]})])

    marg_factor.add_edges_from([
        ('cov_factor','cov_factor_T',{'type':'arg'}),
        ('cov_factor_T','X',{'type':'param','param':'cov_factor'}),
        ('cov_diag','X',{'type':'param','param':'cov_diag'}),
        ('loc','X',{'type':'param','param':'loc'})
        ])
    
    marg_factor.remove_nodes_from(['Wz','z_T','diag']+list(nx.algorithms.dag.ancestors(marg_factor,'z_T')))
    
    return marg_factor


def mixture_from_marg_factor(marg_factor):
    mixture = marg_factor.copy()
    # From among nodes going into the observed node (e.g. mean, cov_factor, cov_diag for factor model), 
    # select between one and all that won't be shared (e.g. if mean is selected, cov_factor and cov_diag will be shared)
    # For now, select all

    parents_of_X = list(marg_factor.predecessors('X'))
    nodes_to_add_to_new_plate = nx.algorithms.ancestors(mixture, 'X')

    # Put the selected nodes, and all their ancestors, onto a new plate, which is indexed
    for node in nodes_to_add_to_new_plate:
        if 'plates' in mixture.nodes[node]:
            mixture.nodes[node]['plates'] = ['C'] + mixture.nodes[node]['plates']
        else:
            mixture.nodes[node]['plates'] = ['C']
        if 'shape' in mixture.nodes[node]:
            mixture.nodes[node]['shape'] = mixture.nodes[node]['shape']+'C'
    # Need to generate init statements for all the params, with plate index as id
    nodes = [
        # Add a dirichlet mixing proportions node, with a concentration parameter parent.
        ('mixing_proportions_concentration',{'type':'param','shape':'C','constraint':'positive'}),
        ('mixing_proportions',{'distribution':dist.Dirichlet,'type':'latent'}),
        # Add a categorical assignment node to the observation plate, child of the dirichlet node, with an infer={'enumerate': 'parallel'} attribute.
        ('assignment',{'distribution':dist.Categorical,'type':'latent','plates':['N']})
    ]
    # Add an indexing node per parent of X
    for node in parents_of_X:
        nodes.append(
            (node+'_idx', {'type':'function', 'function':torch.index_select, 'args':('t', Num(0), 'p'), 'plates':['N']})
        )
    edges = [
        ('mixing_proportions_concentration','mixing_proportions',{'type':'param','param':'concentration'}),
        ('mixing_proportions','assignment',{'type':'param','param':'probs'}),
    ]
    for node in parents_of_X:
        # from node to indexed node
        edges.append((node,node+'_idx',{'type':'arg'}))
        # from assignment to indexed node
        edges.append(('assignment',node+'_idx',{'type':'arg'}))
        # from indexed node to X
        edges.append((node+'_idx','X',mixture.edges[(node,'X')]))
        mixture.remove_edge(node,'X')
        
    nodes.extend([
        ('loc_loc',{'type':'param','shape':'DC'}),
        ('loc_scale',{'type':'param','shape':'DC','constraint':'positive'}),
        ('loc',{'distribution':dist.Normal,'type':'latent'})
    ])

    edges.extend([    
        ('loc_loc','loc',{'type':'param','param':'loc'}),
        ('loc_scale','loc',{'type':'param','param':'scale'}),
        ])

    mixture.add_nodes_from(nodes)
    mixture.add_edges_from(edges)

    del mixture.nodes['loc']['function']
    del mixture.nodes['loc']['args']
    # add the new plate dim to the shapes

    remove_nodes_keep_edges(mixture, 'cov_factor_T')

    return mixture