import numpy as np
import concurrent.futures
from collections import OrderedDict
from pathlib import Path
import sys
import csv
import time
from main import *

N_TRY = 20
MAX_STEP = 10000000
MAX_TIME = 60
n50 = Path("./n50")
n50template = "uf50-0{:d}.cnf"


models = OrderedDict([
    ("amoeba@0.275", [AmoebaSAT, {"error_rate":0.275}]),
    #("amoeba@0.25", [AmoebaSAT, {"error_rate":0.25}]),
    #("amoeba@0.09", [AmoebaSAT, {"error_rate":0.09}]),
    #("walk", [WalkSAT, {}]),
    #("amoebaWhite", [AmoebaWhite, {"error_rate":0.3}]),
    #("amoebaWhite@0.1", [AmoebaWhite, {"error_rate":0.1}]),
    #("amoebaWhiteConstr@0.26", [AmoebaWhiteConstr, {"error_rate":0.26}]),
])

def worker(args):
    num, probNo, prob, m, kwargs = args
    np.random.seed(num + 1337)
    model = m(prob, **kwargs)
    t0 = time.time()
    steps = 0
    times = 0
    #print("prob", probNo, ", seed", num, "begin")
    for ite in range(MAX_STEP):
        #if ite % 10000 == 0: sys.stdout.write("\r{}/{} {} {:%}".format(t, N_TRY, steps[t-1] if t>0 else -1, ite/MAX_STEP))
        if model.check():
            steps = ite
            break
        model.step()
    else:
        steps = MAX_STEP
    t1 = time.time()
    print("prob", probNo, ", seed", num, "done")
    times = t1 - t0
    return (steps, times)


if __name__ == "__main__":
    for key, (m, kwargs) in models.items():
        cfile = open("{}_stat.csv".format(key), "w")
        writer = csv.writer(cfile)
        writer.writerow(["probNo", "try_n", "step_mean", "step_std", "time_mean", "time_std"])

        job = []
        for i in range(1,21):
            fn = n50 / n50template.format(i)
            prob = []
            with fn.open() as f:
                for line in f:
                    if line[0] == "c" or line[0] == "p": continue
                    if line[0] == "%": break
                    items = line.lstrip().split()[:3]
                    prob.append(list(map(int, items)))
            job.extend(list(zip(range(N_TRY), [i]*N_TRY, [prob]*N_TRY, [m]*N_TRY, [kwargs]*N_TRY)))
        print("job length:", len(job))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            working = executor.map(worker, job)
            for i in range(1,21):
                rets = [next(working) for _ in range(N_TRY)]
                steps, times = zip(*rets)
                steps = np.array(steps)
                times = np.array(times)
                print("No. {}, steps: {} +- {}, time: {} +- {}".format(i, steps.mean(), steps.std(), times.mean(), times.std()))
                writer.writerow([i, N_TRY, steps.mean(), steps.std(), times.mean(), times.std()])
