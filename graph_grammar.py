def marginalize_factor(factor_DAG):
    marg_factor = factor_DAG.copy()
    
    marg_factor.nodes['X']['distribution'] = dist.LowRankMultivariateNormal
    marg_factor.add_nodes_from([('loc',{'type':'function', 'function':torch.zeros, 'args':(Name(id='D'),)}),])
    marg_factor.add_nodes_from([('cov_factor_T',{'type':'function', 'function':torch.transpose, 'args':(Num(0),Num(1))})])

    marg_factor.add_edges_from([
        ('cov_factor','cov_factor_T',{'type':'arg'}),
        ('cov_factor_T','X',{'type':'param','param':'cov_factor'}),
        ('cov_diag','X',{'type':'param','param':'cov_diag'}),
        ('loc','X',{'type':'param','param':'loc'})
        ])
    
    marg_factor.remove_nodes_from(['Wz','z_T','diag']+list(nx.algorithms.dag.ancestors(marg_factor,'z_T')))
    
    return marg_factor

