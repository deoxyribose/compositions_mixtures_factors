import os
from train_job import *

def run_next_job(job_list):
	for job in job_list:
		K, restart, init = job
		_id = '_'.join([str(K), str(restart), init])
		if os.path.exists(_id + '.p') or os.path.exists(_id + 'started'):
			continue
		train_job(*job)