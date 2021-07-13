import scipy.optimize, scipy.integrate
from math import sqrt

def d1_metric(f,g,bounds=[-float("inf"),float("inf")]):
    def F(x): return abs(f(x)-g(x))
    d1 = scipy.integrate.quad(F,bounds[0],bounds[1])
    return d1[0]

def d2_metric(f,g,bounds=[-float("inf"),float("inf")]):
    def F(x): return (f(x)-g(x))**2
    d2 = scipy.integrate.quad(F,bounds[0],bounds[1])
    return sqrt(d2[0])

def sup_metric(f, g, whereMax=False):
    def F(x): return -abs(f(x)-g(x))
    sup = scipy.optimize.minimize_scalar(F)
    if whereMax: print(sup.x)
    try:
        return abs(sup.fun)[0]
    except:
        return abs(sup.fun)

# # TEST
# from GMM import norm
# import numpy as np
# import matplotlib.pyplot as plt

# f = lambda x: norm(x,0,1)
# g = lambda x: norm(x,2,2)

# x = np.linspace(-5,10,1000)
# plt.plot(x,f(x))
# plt.plot(x,g(x))

# d1 = d1_metric(f,g)
# d2 = d2_metric(f,g)
# sup = sup_metric(f,g,whereMax=1)


# plt.show()


