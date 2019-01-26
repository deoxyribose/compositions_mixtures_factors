import torch
import numpy as np

def make_normal_data(N,D):
	some_data_locs = np.random.randn(D)*10
	some_data_scales = np.sqrt((np.random.randn(D)*10)**2)
	data = np.random.randn(N,D)*some_data_scales + some_data_locs
	data = np.float32(data) # float64 creates mismatch with torch's defaults
	return torch.tensor(data)

def make_categorical_data(N,D):
	# number of categories is sampled between 2 and number of features
	C = np.random.choice(np.arange(2,D*2),D)
	print(C)
	categorical_matrix = []
	for c in C:
		categorical_matrix.append(np.random.choice(c, size=N))
	data = np.array(categorical_matrix).T
	data = np.float32(data) # float64 creates mismatch with torch's defaults
	return torch.tensor(data)
