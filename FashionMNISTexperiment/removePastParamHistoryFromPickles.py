import numpy as np
import pickle
from os import listdir, remove
from os.path import isfile, join
import sys
sys.path.append("..")
sys.path.append("../factor/")

paths = ["./"]
pickle_jar = [path+f for path in paths for f in listdir(path) if isfile(join(path, f)) if f.endswith('.p')]

for p in pickle_jar:
    with open(p, "rb") as f:
        restart,inference_results = pickle.load(f)
    for restart in range(len(inference_results)):
        trace, losses, lppds, param_history, init, time = inference_results[restart]
        # save just last values, since that's all we need for incremental inference, and the rest fills too much space
        param_history = dict(zip(param_history.keys(), map(lambda x: x[-1], param_history.values())))
        inference_results[restart] = trace, losses, lppds, param_history, init, time
    remove(p)
    with open(p, "wb") as f:
        pickle.dump((restart,inference_results), f)