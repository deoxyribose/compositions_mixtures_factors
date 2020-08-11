import numpy as np

def train_test_split(X, proportion_of_data_for_testing = 0.1):
    N,_ = X.shape
    test_idxs = np.random.choice(N,size=int(N*proportion_of_data_for_testing),replace=False)
    mask = np.ones(N,dtype=bool)
    mask[test_idxs] = False 
    data = X[mask]
    test_data = X[~mask]
    return data, test_data