import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil, log, e, pi
from scipy import stats, integrate
from statsmodels.distributions.empirical_distribution import ECDF


##generating the distributions
gauss1 = []
gauss2 = []

for i in range(10000):
    gauss1.append(random.gauss(30,5))
    gauss2.append(random.gauss(30,10))

mu = np.mean(gauss1)
sigma = np.var(gauss1)**0.5


##optimal bin size
def sirina(variance, N):
    return 3.49*(sigma**2)*(N**(-1/3))

h = np.arange(min(gauss1),max(gauss1),sirina(sigma,len(gauss1)))
##h = np.arange(0,60,5/6)


##maximum bin
def tallest_bin(heights, values):
    maximum_index = list(heights).index(max(list(heights)))
    return 'Interval: {}, Vrednost: {}'.format((round(values[maximum_index],2), round(values[maximum_index+1],2)), round(y[maximum_index],4))


## histrogram and density of the gauss1 set
histogram1 = plt.subplot(221)
y, x, _ = plt.hist(gauss1, bins=h, density=True)
plt.grid(True)
plt.title('Histogram in denziteta')

pdf1 =  stats.gaussian_kde(gauss1)
x1 = np.linspace(floor(min(gauss1)),ceil(max(gauss1)), 10000)
plt.plot(x1, pdf1(x1))


##this line set the grid size, but when you zoom the histogram/plot, the grid does not change
##histogram.set_xticks(np.arange(floor(min(gauss1))-5,ceil(max(gauss1))+5,1))


## histrogram and density of the gauss2 set
histogram2 = plt.subplot(222, sharex=histogram1, sharey=histogram1)
y, x, _ = plt.hist(gauss2, bins=h, density=True)

pdf2 = stats.gaussian_kde(gauss2)
x2 = np.linspace(floor(min(gauss2)),ceil(max(gauss2)), 10000)
plt.plot(x2, pdf2(x2), color='red')
plt.grid(True)


##Distributions of both sets
ecdf1 = ECDF(gauss1)

distribucija1 = plt.subplot(223, sharex=histogram1)
plt.plot(x1, ecdf1(x1))
plt.grid(True)
plt.title('Distribucija')

ecdf2 = ECDF(gauss2)

distribucija2 = plt.subplot(224, sharex=histogram2)
plt.plot(x2, ecdf2(x2), color='red')
plt.grid(True)

plt.subplots_adjust(hspace=0.5)


plt.show()


##x = np.linspace(stats.norm.ppf(0.01,30,5), stats.norm.ppf(0.99,30,5))
##
##plt.plot(x, stats.norm.pdf(x,30,5))

