import random as r
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from renyi import JRD_pdfs, generalised_RD

r.seed(0)

koliko = 30

gausses = {}
for i in range(koliko):
    a=[]
    for j in range(10000):
        a.append(r.gauss(i,2))
    gausses[i] = a

hists = {}
for i in range(koliko):
    hists[i] = np.histogram(gausses[i],bins=15, density=1)

JRD1 = []
for i in range(koliko):
    JRD1.append(generalised_RD([hists[0], hists[i]], 0.5))

pdfs = {}
for i in range(koliko):
    pdfs[i] = stats.gaussian_kde(gausses[i])

JRD2 = []
for i in range(koliko):
    print(i)
    JRD2.append(JRD_pdfs([pdfs[0], pdfs[i]], 0.5))

plt.subplot(131)
plt.plot(range(koliko), JRD1, label="po histogramih")
plt.plot(range(koliko), JRD2, label="po pdf")
plt.grid(1)
plt.legend()

plt.subplot(132)
plt.plot(np.linspace(-10,30,1000), pdfs[0](np.linspace(-10,30,1000)), label="0")
plt.plot(np.linspace(-10,30,1000), pdfs[15](np.linspace(-10,30,1000)), label="15")
plt.grid(1)
plt.legend()

plt.subplot(133)
plt.hist(gausses[0], bins = 15, density=1, label="0")
plt.hist(gausses[15], bins=15, density=1, label="15")
plt.grid(1)
plt.legend()

plt.show()




