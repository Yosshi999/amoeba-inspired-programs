import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
np.random.seed(0)

def distance(u, v):
    return np.linalg.norm(u-v)

def fQ(q):
    return np.abs(q)**gamma / (1 + np.abs(q)**gamma)

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("pickle", type=str)
parser.add_argument("--output", type=str, default="animation.gif")
parser.add_argument("--frames", type=int, default=100)
args = parser.parse_args()
import cv2
im = cv2.resize(cv2.imread(args.input), (200, 200))
mask = np.all(im != 0, axis=2).astype(np.int)


lines = None
points = None
fig,ax = plt.subplots(figsize=(3,3))
def init():
    global lines
    global points
    ax.set_title("t={}".format(ite))
    ax.set_xlim(0, 200)
    ax.set_ylim(200, 0)
    ax.imshow(mask)
    plt.axis("equal")

    lines = dict()
    for i in range(len(V)):
        for e in E[i]:
           if i < e:
                im, = ax.plot(*V[[i,e]].T, "r-", linewidth=D[(i,e)] * 10)
                lines[(i, e)] = im
    points = dict()
    for c in C:
        im, = ax.plot(*V[c].T, "bo")
        points[c] = im
    return tuple(lines.values()) + tuple(points.values())
    
def update():
    global lines
    global points
    ax.set_title("t={}".format(ite))
    for k in lines.keys():
        lines[k].set_linewidth(D[k] * 10)
        
    
    #for c in C:  # show source/sink
    #    if c == s or c == t:
    #        points[c].set_color("r")
    #    else:
    #        points[c].set_color("b")
    return tuple(lines.values()) + tuple(points.values())



with open(args.pickle, "rb") as f:
    V, E, C = pickle.load(f)
V = np.array(V) # nodes
C = np.array(C) # cities
D = dict()      # Dij
initD = 0.01
I = 2.0
gamma = 1.8
h = 0.03        # Euler
for i, e in enumerate(E):
    for ei in e:
        D[(i, ei)] = initD

print("V:", len(V))

s,t =None, None
ite = 0
k = dict()

def f(frame):
    global ite
    global s
    global t
    global D
    global k
    if frame * 5 == ite:
        return update()

    mat = np.zeros((len(V), len(V)))
    b = np.zeros(len(V))
    for offset in range(5):
        ite += 1
        mat[:] = 0
        b[:] = 0

        # forward
        for (i,j), d in D.items():
            k[(i,j)] = d / distance(V[i], V[j])

        s, t = np.random.choice(C, 2, replace=False)
        for i in range(len(V)):
            if i==s:
                for e in E[i]:
                    mat[i, e] = -k[(i,e)]
            else:
                ksum = 0
                for e in E[i]:
                    mat[i, e] = -k[(i,e)]
                    ksum += k[(i,e)]
                mat[i,i] = ksum
                
        b[s] = I
        b[t] = -I
        P = np.linalg.solve(mat, b)
        Q = dict()
        for (i,j), v in k.items():
            Q[(i,j)] = v * (P[i] - P[j])

        # update
        for (i,j), d in D.items():
            D[(i,j)] = d + h * (fQ(Q[(i,j)]) - d)

        print("iteration {:d}, flows {:d}->{:d}, Dmin {:f}, Dmax {:f}".format(ite, s,t, min(D.values()), max(D.values())))
    return update()

anim = FuncAnimation(fig, f, init_func=init, blit=True, frames=args.frames)
anim.save(args.output, writer='imagemagick', fps=5)

exit(0)






