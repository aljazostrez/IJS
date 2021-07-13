import numpy as np
import matplotlib.pyplot as plt
from math import log, e
import random
from renyi import *
from scipy import stats

def renyi_gauss(mu_j, sigma_j, mu_i, sigma_i, alpha):
    if alpha != 1:
        var_zvezdica = alpha*(sigma_j ** 2) + (1-alpha)*(sigma_i**2)
        rezultat = log(sigma_j/sigma_i, e) + (1/(2*(alpha-1)))*log((sigma_j**2)/var_zvezdica, e) + (alpha*((mu_i - mu_j)**2))/(2*var_zvezdica)
        return rezultat
    else:
        rezultat = (1/(2*(sigma_j**2)))*((mu_i - mu_j)**2 + sigma_i**2 - sigma_j**2) + log(sigma_j/sigma_i, e)
        return rezultat

mu0 = 30
mu1 = 30
sigma0 = 5
sigma1 = 10

g1 = []
g2 = []
for i in range(10000):
    g1.append(random.gauss(30,5))
    g2.append(random.gauss(30,10))

pdf1 = stats.gaussian_kde(g1)
pdf2 = stats.gaussian_kde(g2)

x=np.linspace(min(g2), max(g2), 1000)
plt.plot(x, pdf1(x), label='mu=30, var=5')
plt.plot(x, pdf2(x), label='mu=30, var=10')
plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.legend()
plt.grid(1)
plt.show()

renyi1 = []
renyi4 = []
for i in np.arange(0.1,3,0.1):
    renyi1.append(renyi_divergence_cont(pdf1,pdf2,i,min(g2),max(g2)))
    renyi4.append(renyi_gauss(mu1,sigma1,mu0,sigma0,i))


plt.plot(np.arange(0.1,3,0.1), renyi1, label='Divergenca glede na približek gostote')  
plt.plot(np.arange(0.1,3,0.1), renyi4, label='Divergenca normalnih porazdelitev')
plt.xlabel('α')
plt.ylabel('Divergenca')
plt.grid(1)
plt.legend()
plt.show()
