import numpy as np
import matplotlib.pyplot as plt
import argparse
from tools.hexmesh import hexagon
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("--output", type=str, default="node.pickle")
args = parser.parse_args()

np.random.seed(0)
_p, _e = hexagon(5)
_p += np.random.randn(len(_p), 3) * 0.02
p = _p * 35 + 70
p = p.astype(np.int)

import cv2
im = cv2.resize(cv2.imread(args.input), (200, 200))
mask = np.all(im != 0, axis=2).astype(np.int)
mask2 = ((im[:,:,0] > 100) * (im[:,:,2] < 100)).astype(np.uint8)
mask2 = cv2.dilate(mask2, np.ones((3,3), np.uint8))
_, contours, _ = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
M = [cv2.moments(cnt) for cnt in contours]
city = np.array(list(map(lambda m: [int(m['m10']/m['m00']), int(m['m01']/m['m00'])], M)))
print("cities:", len(city))

keepe = []
for i,x in enumerate(_e):
    for c in x:
        if np.any(p[c,:2] < 0) or np.any(p[c,:2] >= 200) or mask[p[c,1], p[c,0]] == 0:
            break
    else:
        keepe.append(i)
e = _e[keepe]
keepIdx = np.unique(e)
keepP = p[keepIdx]

print("V:", len(keepP))

cityIdx = []
cityIdx_keep = []
for c in city:
    dist = np.sum((keepP[:,:2] - [c[0], c[1]]) ** 2, axis=1)
    am = np.argmin(dist)
    cityIdx_keep.append(am)
    cityIdx.append(keepIdx[am])
#print(np.array(cityIdx))


E = []
for i in keepIdx:
    idx = np.any(e == i, axis=1)
    neighbor = np.unique(e[idx])
    E.append(np.where(np.any(keepIdx.reshape(1,-1) == neighbor.reshape(-1,1), axis=0) * (keepIdx != i))[0].tolist())

import pickle
with open(args.output, "wb") as f:
    pickle.dump((keepP[:,:2].tolist(), E, cityIdx_keep), f)

plt.title("t=0")
plt.axis("equal")
plt.xlim(0, 200)
plt.ylim(200, 0)
for x in e:
    plt.plot(*p[[*x, x[0]], :2].T, "r-", linewidth=0.2)
plt.plot(*p[cityIdx, :2].T, "bo")
#plt.imshow(im[:,:,::-1])
plt.imshow(mask)

plt.show()
