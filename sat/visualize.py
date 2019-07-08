import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MAX_STEP = 10000000
ERR = False

names = [
    "amoeba@0.25",
    "amoeba@0.09",
    #"walk",
    "amoebaWhiteConstr@0.26",
    #"amoebaWhite@0.3",
    "amoebaWhite@0.1"
]
dfs = [pd.read_csv(n + "_stat.csv") for n in names]

rank = np.argsort(dfs[2]["step_mean"].values)
x = np.zeros(20)
for i, r in enumerate(rank):
    x[r] = i + 1
print(x)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))


#plt.ylim(100,3e7)
plt.yscale("log")
plt.title("steps")
plt.xlabel("prob No.")
plt.xticks(np.arange(1, 21), rank+1)
plt.ylabel("steps")

for i,df in enumerate(dfs):
    if ERR:
        plt.errorbar(x + i*0.05, df["step_mean"], fmt="o", yerr=df["step_std"], label=names[i])
    else:
        if i < 2: c="x"
        elif i==2: c="o"
        else: c="."
        plt.errorbar(x, df["step_mean"], fmt=c, label=names[i])
#plt.plot(x, np.repeat(MAX_STEP, len(x)), "-", label="maxstep")
plt.legend(loc="upper left")


plt.subplot(1,2,2)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

plt.yscale("log")
plt.title("time")
plt.xlabel("prob No.")
plt.xticks(np.arange(1, 21), rank+1)
plt.ylabel("sec")
for i,df in enumerate(dfs):
    if ERR:
        plt.errorbar(x+0.05*i, df["time_mean"], fmt="o", yerr=df["time_std"], label=names[i])
    else:
        if i < 2: c="x"
        elif i==2: c="o"
        else: c="."
        plt.errorbar(x, df["time_mean"], fmt=c, label=names[i])
plt.legend()
plt.show()


