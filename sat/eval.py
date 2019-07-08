import numpy as np
from collections import OrderedDict
from pathlib import Path
import sys
import csv
import time
from main import *

np.random.seed(0)
N_TRY = 100
MAX_STEP = 10000000
MAX_TIME = 60
n50 = Path("./n50")
n50template = "uf50-0{:d}.cnf"


models = OrderedDict([
    ("amoeba@0.25", [AmoebaSAT, {"error_rate":0.25}]),
    ("amoeba@0.09", [AmoebaSAT, {"error_rate":0.09}]),
    ("walkSAT", [WalkSAT, {}]),
])

for key, (m, kwargs) in models.items():
    cfile = open("{}_stat.csv".format(key), "w")
    writer = csv.writer(cfile)
    writer.writerow(["probNo", "try_n", "step_mean", "step_std", "time_mean", "time_std"])

    for i in range(1,21):
        fn = n50 / n50template.format(i)
        prob = []
        with fn.open() as f:
            for line in f:
                if line[0] == "c" or line[0] == "p": continue
                if line[0] == "%": break
                items = line.lstrip().split()[:3]
                prob.append(list(map(int, items)))
        print("No. {}:".format(i))
        steps = np.zeros(N_TRY)
        times = np.zeros(N_TRY)
        for t in range(N_TRY):
            model = m(prob, **kwargs)
            t0 = time.time()
            for ite in range(MAX_STEP):
                if ite % 10000 == 0: sys.stdout.write("\r{}/{} {} {:%}".format(t, N_TRY, steps[t-1] if t>0 else -1, ite/MAX_STEP))
                if model.check():
                    steps[t] = ite
                    break
                model.step()
            else:
                steps[t] = MAX_STEP
            t1 = time.time()
            times[t] = t1 - t0
        writer.writerow([i, N_TRY, steps.mean(), steps.std(), times.mean(), times.std()])
        print()
