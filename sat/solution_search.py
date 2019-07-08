import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main import *
np.random.seed(0)

err = float(sys.argv[1])
inst = [
    [-1, 3, -4],
    [1, -2, 3],
    [1,2,3],
    [1,2,4],
    [1,-3,-4],
    [-2,3,4],
    [2,-3,-4],
    [2,-3,4],
    [2,3,4]]

answer = [[1,1,1,1], [1,1,1,0], [0,1,1,0]]
col = "rgb"
model = AmoebaSAT(inst, err)
Xs = []
ans = ""
valid = 0

for i in range(2000):
    if model.check():
        a = model.assign
        aa = np.zeros(len(a))
        aa[a] = 1
        c = col[answer.index(aa.tolist())]
        ans += c
        valid += 1
    else:
        ans += "."
    Xs.append(model.X.reshape(-1).copy())
    model.step()
Xs = np.array(Xs)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="3d")
ax.set_ylim(0, 2000)
ax.set_zlim(-1, 20)
for i in range(Xs.shape[1]):
    ax.plot(np.repeat(i*5, len(Xs)), np.arange(len(Xs)), "-", lw=0.3, zs=Xs[:,i])

ans += "."
print(ans)
prev_a = "."
begin = 0
for i,a in enumerate(ans):
    if prev_a != a:
        print(a, prev_a)
        if prev_a != ".":
            print(begin, i, a)
            X,Y = np.meshgrid(np.arange(0, Xs.shape[1]*5), np.arange(begin,i))
            Z = np.zeros_like(X) - 1
            ax.plot_surface(X, Y, Z, alpha=0.3, color=prev_a)
        begin = i
        prev_a = a
            
#ax.set_xticks(np.arange(0,Xs.shape[1]*5,5), [r"$X_{%d,%d}$" % (i//2, i%2) for i in range(Xs.shape[1])])
ax.set_xticklabels([r"$X_{%d,%d}$" % (i//2+1, i%2) for i in range(Xs.shape[1])])
ax.set_zticks([-1,0,1])
plt.show()
        
