import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
np.random.seed(0)

def distance(u, v):
    return np.linalg.norm(u-v)

def fQ(q):
#    return np.abs(q)**gamma / (1 + np.abs(q)**gamma)
    qg = q.power(2).power(gamma/2)
    return -(-qg.log1p()).expm1()

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
                im, = ax.plot(*V[[i,e]].T, "r-", linewidth=Dmat[i,e] * 10)
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
    D = Dmat.tocoo()
    for i,e,v in zip(D.row, D.col, D.data):
        if i < e:
            lines[(i,e)].set_linewidth(v * 10)
        
    
    return tuple(lines.values())
    #for c in C:  # show source/sink
    #    if c == s or c == t:
    #        points[c].set_color("r")
    #    else:
    #        points[c].set_color("b")
    #return tuple(lines.values()) + tuple(points.values())



with open(args.pickle, "rb") as f:
    V, E, C = pickle.load(f)
V = np.array(V) # nodes
C = np.array(C) # cities
Dmat = np.zeros((len(V), len(V))) # Dij
invDist = np.zeros_like(Dmat)
edges = []

initD = 0.01
I = 2.0
gamma = 1.8
h = 0.03        # Euler
for i, e in enumerate(E):
    for ei in e:
        Dmat[i, ei] = initD
        invDist[i, ei] = 1 / distance(V[i], V[ei])
        if i < ei: edges.append((i, ei))
Dmat = scipy.sparse.csr_matrix(Dmat)
invDist = scipy.sparse.csr_matrix(invDist)

print("V:", len(V))

s,t =None, None
ite = 0

def f(frame):
    global ite
    global s
    global t
    global Dmat
    if frame * 5 == ite:
        return update()

    b = scipy.sparse.lil_matrix(np.zeros((len(V), 1)))
    for offset in range(5):
        ite += 1
        b[:] = 0

        # forward
        mat = Dmat.multiply(invDist)

        s, t = np.random.choice(C, 2, replace=False)
        
        mat.setdiag(-np.asarray(mat.sum(axis=1)).reshape(-1))

        b[s] = I
        b[t] = -I
        P = scipy.sparse.linalg.spsolve(mat, b.tocsr())

        Q = mat.multiply(P[:,None]) - mat.multiply(P[None,:])

        # update
        Dmat = Dmat + h * (scipy.sparse.csr_matrix(fQ(Q)) - Dmat)
        #print(type(Dmat))

        print("iteration {:d}, flows {:d}->{:d}, Dmin {:f}, Dmax {:f}".format(ite, s,t, Dmat.data.min(), Dmat.max()))
    Dmat.eliminate_zeros()
    return update()

anim = FuncAnimation(fig, f, init_func=init, blit=True, frames=args.frames)
anim.save(args.output, writer='imagemagick', fps=5)

exit(0)
