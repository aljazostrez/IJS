import time
from renyi import JRD_pdfs, renyi_entropy_cont
import random as r
import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
from math import pi, e, log
from tabulate import tabulate


## definicija normalne
def norm(x, mu=1, sigma=1):
    return (1/((2*pi*sigma**2)**0.5)) * e ** (-(x-mu)**2/(2*sigma**2))

## fiksna normalna, s katero bomo primerjali vse ostale
def normalna0(x): return norm(x, 0, 1)



## definicija konkretnih normalnih za primerjavo
normalne = []

range_mu = np.arange(0,5,0.5)
range_sigma = np.arange(-10000,5,0.5)

for i in range_mu:
    normalne_mu = []
    for j in range_sigma:
        def normalna(x): return norm(x,mu=i,sigma=j)
        normalne_mu.append(normalna)
    normalne.append(normalne_mu)

# ## izracun JRD pri redu alfa=0.5
# JRD = []
# for i in range(len(normalne)):
#     JRD_i = []
#     JRD_i.append("mu={}".format(range_mu[i]))
#     for j in range(len(normalne[i])):
#         # print(j)
#         JRD_i.append(JRD_pdfs([normalna0, normalne[i][j]], 0.5))
#     JRD.append(JRD_i)
#     # print(i)


## glava tabele
# header = []
# header.append("")
# for i in range_sigma:
#     header.append("sigma={}".format(i))

# ## izris tabele
# print(tabulate(JRD, headers=header, tablefmt='orgtbl'))


# ## generacija podatkov za kasneje, ce bo treba
# data = []

# for i in np.arange(0,5,0.5):
#     data_mu = []
#     for j in np.arange(0.5,5,0.5):
#         data_mu_sigma = []
#         for _ in range(10000):
#             data_mu_sigma.append(r.gauss(i,j))
#         data_mu.append(data_mu_sigma)
#     data.append(data_mu)

# normalne = []
# for i in range(len(data)):
#     normalne_i = []
#     for j in range(len(data[i])):
#         normalne_i.append(stats.gaussian_kde(data[i][j]))
#     normalne.append(normalne_i)


# x = np.linspace(-10,10,1000)
# for i in range(len(normalne)):
#     plt.figure()
#     for j in range(len(normalne[i])):
#         try:
#             plt.plot(x, normalne[i][j](x))
#         except: print(i,j)    
#     print(i)
x = np.linspace(-10,10,10000)
plt.plot(x, normalne[0][0](x))
plt.show()

print(renyi_entropy_cont(normalna0, 0.5, -6,4))