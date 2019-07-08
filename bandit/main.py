import numpy as np
import matplotlib.pyplot as plt

class ToW:
    def __init__(self, alpha=1., mu=1.):
        self.alpha = alpha
        self.mu = mu
        self.initialize()

    def initialize(self):
        self.x = np.zeros(2)
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        self.s = 0.
        self.ss = np.zeros(2)
        self.q = 0.
        self.r = 0.

    def update(self, choice, light):
        nx = self.x + self.v
        ns = self.s - self.v.sum()
        #self.s = -self.x.sum()

        nr = self.mu * ((choice[0] - 2 * light[0]) - (choice[1] - 2 * light[1]))
        nq = nr + self.alpha * self.q
        nss = np.array([ns + nq, ns-nq])
        #nss = ns + [self.q, -self.q]
        
        na = light  * (-1)*np.array(nss<=0).astype(np.int) + (1-light) * np.array(nss>=0).astype(np.int)
        #na = light  * (-1)  + (1-light)
        #print(choice, light, nr, na)
        #na = light * (-1)*(nss<=0).astype(np.int) + (1-light) * (nss>=0).astype(np.int)
        nv = self.v + na


        self.x[:] = nx
        self.v[:] = nv
        self.a[:] = na
        self.s = ns
        self.ss[:] = nss
        self.q = nq
        self.r = nr

    def act(self):
        return self.x > 0
        #return self.x > self.x[::-1]

def test():
    np.random.seed(0)
    #p = np.array([.7, .3])
    #plt.title("P = " + str(p) + "->" + str(p[::-1]) + "at t=400")
    p = np.array([.45, .55])
    plt.title("P = " + str(p))
    plt.grid(True)
    #model = ToW(alpha=0.999)
    model = ToW()
    xa, xb, s, q = [], [], [], []
    N = 800
    for i in range(N):
        #if i == 400: p = p[::-1]
        xa.append(model.x[0])
        xb.append(model.x[1])
        s.append(model.s)
        q.append(model.q)

        aa = model.act()
        choice = np.zeros(2)
        choice[aa] = 1
        light = (np.random.rand(2) < 1 - p).astype(np.int) * choice
        model.update(choice, light)
        
    
    plt.plot(np.arange(N), xa, label="Xa")
    plt.plot(np.arange(N), xb, label="Xb")
    plt.plot(np.arange(N), s, label="S")
    plt.plot(np.arange(N), q, label="Q")
    plt.legend()
    plt.show()
        
def test2():
    np.random.seed(0)
    p = np.array([.7, .3])
    plt.title("P = " + str(p) + "->" + str(p[::-1]) + "at t=400")
    plt.grid(True)
    model = ToW(alpha=0.999)
    xa, xb, s, q = [], [], [], []
    N = 800
    for i in range(N):
        if i == 400: p = p[::-1]
        xa.append(model.x[0])
        xb.append(model.x[1])
        s.append(model.s)
        q.append(model.q)

        aa = model.act()
        choice = np.zeros(2)
        choice[aa] = 1
        light = (np.random.rand(2) < 1 - p).astype(np.int) * choice
        model.update(choice, light)
        
    
    plt.plot(np.arange(N), xa, label="Xa")
    plt.plot(np.arange(N), xb, label="Xb")
    plt.plot(np.arange(N), s, label="S")
    plt.plot(np.arange(N), q, label="Q")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()
    test2()
