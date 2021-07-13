from math import pi, cos, sin, e, sqrt
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

## histogram -> pdf
def hist_to_fun(histogram):
    # histogram: nabor oblike (heights, edges) - stolpci in meje stolpcev histograma
    # RETURN: funkcija, ki ustreza argumentu histogram
    
    def nested(x, hist):
        [heights, edges] = hist
        # kjer ima histogram vrednost 0, mu bomo zaradi racunanja v nadaljevanju priredili vrednost eps
        h_val = np.finfo(float).eps
        for i in range(len(edges)-1):
            if (edges[i] <= x <= edges[i+1]):
                if heights[i] != 0:
                    h_val = heights[i]
                break
        return h_val
    fun = lambda x: nested(x, histogram)
    return fun

def uniform(x):
    return np.where(abs(x)<1, 1/2, 0)

def triangular(x):
    return np.where(abs(x)<1, 1-abs(x),0)

def epanechnikov(x):
    return np.where(abs(x)<1, 3/4*(1-x**2),0)

def quartic(x):
    return np.where(abs(x)<1, 15/16*(1-x**2)**2, 0)

def triweight(x):
    return np.where(abs(x)<1, 35/32*(1-x**2)**3, 0)

def tricube(x):
    return np.where(abs(x)<1, 70/81*(1-abs(x)**3)**3, 0)

def cosine(x):
    return np.where(abs(x)<1, pi/4*np.cos(pi/2*x), 0)

def gaussian(x):
    return 1/np.sqrt(2*np.pi)*np.e**(-1/2*x**2)

def logistic(x):
    return 1/(np.e**x+2+np.e**(-x))

def sigmoid(x):
    return 2/np.pi * 1/(np.e**x+np.e**(-x))

def silverman(x):
    return 1/2 * np.e**(-np.abs(x)/np.sqrt(2)) * np.sin(np.abs(x)/np.sqrt(2) + np.pi/4)

kernels = {
    "uniform": uniform,
    "triangular": triangular,
    "epanechnikov": epanechnikov,
    "quartic": quartic,
    "triweight": triweight,
    "tricube": tricube,
    "gaussian": gaussian,
    "cosine": cosine,
    "logistic": logistic,
    "sigmoid": sigmoid,
    "silverman": silverman
}

def kde(data, kernel="gaussian"):
    """kernels = uniform, triangular, epanechnikov, quartic,
                 triweight, tricube, gaussian, cosine,
                 logistic, sigmoid, silverman"""

    bandwidth = np.std(data)*(4/(3*len(data)))**(1/5)

    if kernel not in kernels: return ("Kernel does not exist.")
    
    def f(x):
        s = 0
        for d in data:
            s += kernels[kernel]((x-d)/bandwidth)
        return 1/(len(data)*bandwidth)*s

    return f

def kth_nearest(x,k,data):
    lengths = np.abs(np.repeat(x,len(data))-data)
    kth_length = sorted(lengths)[k]
    return kth_length


def adapt_kde(data, k=None, h=None, kernel="gaussian"):
    if k==None:
        k = int(sqrt(len(data)))
        k = int(len(data)/2)
    if h==None: h = np.std(data)*(4/(3*len(data)))**(1/5)

    def f(x):
        s = 0
        for d in data:
            hd = h*kth_nearest(d,k,data)
            s += 1/hd *  kernels[kernel]((x-d)/hd)
        n = len(data)
        return 1/n * s
    return f
# ## TEST
# import random as r

# r.seed(0)

# dat = []
# for i in range(1000):
#     dat.append(r.betavariate(1.1,1))


# pdf1 = adapt_kde(dat)
# pdf2 = kde(dat)
# x = np.linspace(0,1,1000)
# plt.plot(x,pdf1(x))
# plt.plot(x,pdf2(x))
# plt.show()

# pdf = kde(dat,kernel="gaussian")
# x = np.linspace(-20,20,1000)

# plt.plot(x, pdf(x))
# plt.show()


## IZRIS JEDER
x1 = np.linspace(-1,1,1000)
x2 = np.linspace(-10,10,1000)

y1 = [uniform(i) for i in x1]
y2 = [triangular(i) for i in x1]
y3 = [epanechnikov(i) for i in x1]
y4 = [quartic(i) for i in x1]
y5 = [triweight(i) for i in x1]
y6 = [tricube(i) for i in x1]
y7 = [gaussian(i) for i in x2]
y8 = [cosine(i) for i in x1]
y9 = [logistic(i) for i in x2]
y10 = [sigmoid(i) for i in x2]
y11 = [silverman(i) for i in x2]

ax1 = plt.subplot(3,3,1)
plt.plot(x1,y2,label="triangular")
plt.title("triangular")
plt.subplot(3,3,2,sharey=ax1)
plt.plot(x1,y1,label="uniform")
plt.title("uniform")
# plt.legend()

plt.subplot(3,3,3,sharey=ax1)
plt.plot(x1,y3,label="Epanechnikov")
plt.title("Epanechnikov")
plt.subplot(3,3,4,sharey=ax1)
plt.plot(x1,y4,label="quartic")
plt.title("quartic")
# plt.legend()

plt.subplot(3,3,5,sharey=ax1)
plt.plot(x1,y5,label="triweight")
plt.title("triweight")
plt.subplot(3,3,6,sharey=ax1)
plt.plot(x1,y6,label="tricube")
plt.title("tricube")
plt.subplot(3,3,7,sharey=ax1)
plt.plot(x1,y8,label="cosine")
plt.title("cosine")
# plt.legend()

plt.subplot(3,3,8,sharey=ax1)
plt.plot(x2,y7,label="Gaussian")
plt.title("Gaussian")
plt.subplot(3,3,9,sharey=ax1)
plt.plot(x2,y9,label="logistic")
plt.title("logistic")
# plt.legend()

# plt.subplot(3,3,10)
# plt.plot(x2,y10,label="Sigmoid")
# plt.plot(x2,y11,label="silverman")
# plt.legend()

print(min(y1))
print(min(y2))
print(min(y3))
print(min(y4))
print(min(y5))
print(min(y6))
print(min(y7))
print(min(y8))
print(min(y9))
print(min(y10))
print(min(y11))
# edini negativen je silverman

plt.show()