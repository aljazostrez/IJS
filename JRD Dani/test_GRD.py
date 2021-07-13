from renyi import generalised_RD
import random as r
import numpy as np
import matplotlib.pyplot as plt

r.seed(0)

GRD = []
a = []
for i in range(10000):
    a.append(r.gauss(0,2))

hist1 = np.histogram(a, bins=15, density=1)
plt.hist(a, bins=15)

for i in range(200):
    b = []
    for j in range(10000):
        b.append(r.gauss(i,2))
    hist2 = np.histogram(b, bins=15, density=1)
    if i == 14:
        plt.hist(b,bins=15)
    GRD.append(generalised_RD([hist1, hist2], 0.5))

plt.show()

plt.plot(range(200), GRD)
plt.show()