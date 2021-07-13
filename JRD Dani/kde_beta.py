import random as r
from scipy import stats, integrate
import matplotlib.pyplot as plt
import numpy as np

a = []

for i in range(1000000):
    a.append(r.betavariate(4,1))

pdf_beta = stats.gaussian_kde(a)

plt.hist(a, bins=25, density=1, label="Beta (a=4,b=1) dataset")
plt.plot(np.linspace(0,1.5, 1000), pdf_beta(np.linspace(0,1.5,1000)), color='red', label="Pdf po g_kde")
plt.legend()
plt.grid(1)
plt.show()

# print(integrate.quad(pdf_beta,0,1))
