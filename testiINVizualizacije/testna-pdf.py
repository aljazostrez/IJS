import random
from scipy import stats, integrate
import matplotlib.pyplot as plt
from math import pi, e
import numpy as np

with open('samples.txt') as dat:
    g1 = dat.readline().split(',')
    g2 = dat.readline().split(',')
    sample1 = list(map(float, g1))
    sample2 = list(map(float, g2))

x1 = np.linspace(min(sample1) -1, max(sample1)+1, len(sample1))
x2 = np.linspace(min(sample1) -1, max(sample1)+1, len(sample2))

def norm(x, mu, sigma):
        return (1/((2*pi*sigma**2)**0.5)) * e ** (-(x-mu)**2/(2*sigma**2))



GMM1 = 1/3 * norm(x1, 0, 6.2) + 1/3 * norm(x1, 30, 10.3) + 1/3 * norm(x1, 60, 8)
GMM2 = 1/2 * norm(x2, 15, 10.4) + 1/2 * norm(x2, 45, 10.3)

pdf1 = stats.gaussian_kde(sample1)
pdf2 = stats.gaussian_kde(sample2)

plt.plot(x1, GMM1, label='p')
##plt.plot(x1, pdf1(x1), label='epdf1')

plt.plot(x2, GMM2, label='q')
##plt.plot(x2, pdf2(x2), label='epdf2')

plt.legend()
plt.grid(1)
plt.xlabel('x')
plt.ylabel('pdf(x)')

plt.show()
