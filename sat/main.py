#from numba import jit
import numpy as np
from collections import defaultdict
np.random.seed(3)

def _check(problem, assign, M):
    for i in range(M):
        for j in range(3):
            lit = problem[i,j]
            if (lit > 0 and assign[lit-1]) or (lit < 0 and not assign[abs(lit)-1]):
                break
        else:
            # invalid
            return False
    return True

class BaseSAT:
    def __init__(self, problem):
        self.problem = np.zeros((len(problem), 3), np.int)
        for i in range(len(problem)):
            assert len(problem[i]) <= 3
            for j,x in enumerate(problem[i]):
                self.problem[i,j] = x
        vars = set()
        for clause in problem:
            for lit in clause:
                v = abs(lit)
                vars.add(v)
        vars = list(vars)
        assert min(vars) == 1
        assert len(vars) == max(vars) - min(vars) + 1
        self.assign = np.array([False for _ in vars])
        self.vars = vars
        self.clen = len(self.problem)

    def step(self):
        raise NotImplementedError()

    def check(self):
        return _check(self.problem, self.assign, self.clen)
    
class WalkSAT(BaseSAT):
    def __init__(self, problem):
        super().__init__(problem)
        self.assign = np.random.choice([True, False], size=len(self.assign))

    def step(self):
        unsat = []
        for i, clause in enumerate(self.problem):
            for lit in clause:
                if (lit > 0 and self.assign[lit-1]) or (lit < 0 and not self.assign[abs(lit)-1]):
                    break
            else:
                # invalid
                unsat.append(i)
        if len(unsat) == 0: return
        c = self.problem[np.random.choice(unsat)]
        l = np.random.choice(c)
        self.assign[abs(l)-1] = not self.assign[abs(l)-1]

class AmoebaSAT(BaseSAT):
    def __init__(self, problem, error_rate=0.25):
        super().__init__(problem)
        self.negclause = np.zeros((len(self.problem), len(self.vars), 2), np.uint8)
        for i,c in enumerate(self.problem):
            for l in c:
                if l > 0: self.negclause[i, l-1, 0] = 1
                else: self.negclause[i, abs(l)-1, 1] = 1

        contra = []
        for v in self.vars:
            c0 = np.where(self.negclause[:, v-1, 0] == 1)[0]
            c1 = np.where(self.negclause[:, v-1, 1] == 1)[0]
            for c0i in c0:
                for c1i in c1:
                    row = np.zeros((len(self.vars), 2), np.uint8)
                    row |= self.negclause[c0i]
                    row |= self.negclause[c1i]
                    row[v-1, :] = 0
                    if np.sum(np.sum(row, axis=1) >= 2) != 0:
                        # x==1とx==-1がかぶった => Intraに吸収
                        continue
                    row = np.clip(row, 0, 1)
                    contra.append(row)
        self.contra = np.array(contra)
        self.Z = np.random.rand(len(self.vars), 2)
        self.Y = np.zeros((len(self.vars), 2), np.uint8)
        self.L = np.zeros((len(self.vars), 2), np.uint8)
        self.X = np.zeros((len(self.vars), 2), np.int8)
        self.err = error_rate
        
        self.flat_negclause = self.negclause.reshape(-1, len(self.vars)*2)
        self.flat_negclause_sum = self.flat_negclause.sum(axis=1)
        self.flat_contra = self.contra.reshape(-1, len(self.vars)*2)
        self.flat_contra_sum = self.flat_contra.sum(axis=1)

        self.nZ = np.zeros_like(self.Z)
        self.nY = np.zeros_like(self.Y)
        self.nL = np.zeros_like(self.L)
        self.nX = np.zeros_like(self.X)

        self.dot = np.zeros_like(self.flat_negclause_sum)

    def _noise(self):
        return 4 * self.Z * (1-self.Z)
        
    def step(self):
        self.nZ = self._noise()
        self.nL[:] = 0
        # intra
        x1 = (self.X == 1)
        self.nL |= x1[:,::-1]
        #self.nL[:,0] |= np.clip(self.X[:,1], 0, 1).astype(np.uint8)
        #self.nL[:,1] |= np.clip(self.X[:,0], 0, 1).astype(np.uint8)
        # inter
        flat_Xp = x1.reshape(-1)
        #hit = np.sum(
        #        flat_Xp * self.flat_negclause,
        #        axis=1) == self.flat_negclause_sum-1
        np.matmul(flat_Xp, self.flat_negclause.T, out=self.dot)
        hit = (self.dot == self.flat_negclause_sum - 1)
        if np.sum(hit) > 0: self.nL |= (self.X != 1) * np.any(self.negclause[hit], axis=0)
        #hitA = np.sum(
        #        flat_Xp * self.flat_negclause,
        #        axis=1) == self.flat_negclause_sum
        hitA = (self.dot == self.flat_negclause_sum)
        if np.sum(hitA) > 0: self.nL |= np.any(self.negclause[hitA], axis=0)
        # contra
        #hitC = np.sum(
        #         flat_Xp * self.flat_contra,
        #         axis=1) == self.flat_contra_sum
        hitC = flat_Xp @ self.flat_contra.T == self.flat_contra_sum
        if np.sum(hitC) > 0: self.nL |= np.any(self.contra[hitC], axis=0)
        
        self.nY[:] = (self.Z < 1-self.err) * (self.L == 0)

        #self.nX[:] = self.X + (self.Y == 1) - (self.Y == 0)
        self.nX[:] = self.X + self.Y *2 - 1

        self.Z = self.nZ
        self.Y = self.nY
        self.L = self.nL
        np.clip(self.nX, -1, 1, out=self.X)
        return None

    def check(self):
        x1 = self.X == 1
        xn = self.X <= 0
        self.assign[ x1[:,0] * xn[:,1] ] = False
        self.assign[ x1[:,1] * xn[:,0] ] = True
        return super().check()
        #for i,v in enumerate(self.assign):
        #    if self.X[i,0] == 1 and self.X[i,1] <= 0: self.assign[i] = False
        #    if self.X[i,1] == 1 and self.X[i,0] <= 0: self.assign[i] = True
        #if super().check(): return self.assign

class AmoebaWhite(AmoebaSAT):
    def _noise(self):
        return np.random.rand(*self.Z.shape)

class AmoebaWhiteConstr(AmoebaWhite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_cool = np.zeros_like(self.Z)

    def step(self):
        self.nZ = self._noise()
        self.nL[:] = 0
        # intra
        x1 = (self.X == 1)
        self.nL |= x1[:,::-1]
        # inter
        flat_Xp = x1.reshape(-1)
        np.matmul(flat_Xp, self.flat_negclause.T, out=self.dot)
        hit = (self.dot == self.flat_negclause_sum - 1)
        if np.sum(hit) > 0: self.nL |= (self.X != 1) * np.any(self.negclause[hit], axis=0)
        hitA = (self.dot == self.flat_negclause_sum)
        if np.sum(hitA) > 0: self.nL |= np.any(self.negclause[hitA], axis=0)
        # contra
        hitC = flat_Xp @ self.flat_contra.T == self.flat_contra_sum
        if np.sum(hitC) > 0: self.nL |= np.any(self.contra[hitC], axis=0)
        
        self.error_cool = np.clip(self.error_cool - 1, 0, None)
        error_occur = (self.Z >= 1-self.err) * (self.L == 0)
        error_occur[self.error_cool > 0] = False
        self.error_cool[error_occur] += 2
        self.nY[:] = (error_occur != 1) * (self.L == 0)
        

        self.nX[:] = self.X + self.Y *2 - 1

        self.Z = self.nZ
        self.Y = self.nY
        self.L = self.nL
        np.clip(self.nX, -1, 1, out=self.X)
        return None


if __name__ == '__main__':
    inst = [[1,-2], [-2,3,-4], [1,3], [2,-3], [3,-4], [-1,4]]
    model = WalkSAT(inst)
    for i in range(10000):
        if model.check():
            print(i, model.assign)
            break
        model.step()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    model = AmoebaWhite(inst, 0.3)
    Xs = []
    Ls = []
    done = False
    for i in range(10000):
        if i < 30: Ls.append(model.Z.reshape(-1) > 0.75)
        if i < 30: Xs.append(np.copy(model.X.reshape(-1)))
        model.step()
        if not done and model.check():
            print(i, model.assign)
            done = True
    Xs = np.array(Xs)[:30]
    Ls = np.array(Ls)[:30]

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_aspect("equal")
    ax.set_title("amoebaSAT")
    ax.set_zlim(-1, 15)
    ax.set_ylim(0, 30)
    ax.set_ylabel("N")
    for i in range(Xs.shape[1]):
        ax.plot(np.repeat(i*5, len(Xs)), np.arange(len(Xs)), ".-", zs=Xs[:,i])

    ax = fig.add_subplot(122, projection='3d')
    #ax.set_proj_type('ortho')
    ax.set_aspect("equal")
    ax.set_title("Z > 0.75")
    ax.set_zlim(-1, 15)
    ax.set_ylim(0, 30)
    ax.set_ylabel("N")
    for i in range(Ls.shape[1]):
        ax.plot(np.repeat(i*5, len(Ls)), np.arange(len(Ls)), ".-", zs=Ls[:,i])
    plt.show()
