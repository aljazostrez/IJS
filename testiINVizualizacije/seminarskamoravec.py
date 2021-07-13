import random
from scipy import stats, integrate
import matplotlib.pyplot as plt
from math import pi, e
import numpy as np

g1 = []
g2 = []
g3 = []
g4 = []

for i in range(10000):
    g1.append(random.gauss(0,1))
    g2.append(random.gauss(5,1))
    g3.append(random.gauss(0,5))
    g4.append(random.gauss(0,10))

x1 = np.linspace(-25, 25, 10000)
x2 = np.linspace(-25, 25, 10000)
x3 = np.linspace(-25, 25, 10000)
x4 = np.linspace(-25, 25, 10000)


def norm(x, mu, sigma):
        return (1/((2*pi*sigma**2)**0.5)) * e ** (-(x-mu)**2/(2*sigma**2))

def norm1(x): return norm(x, 0, 1)
def norm2(x): return norm(x, 5, 1)
def norm3(x): return norm(x, 0, 5)
def norm4(x): return norm(x, 0, 10)


# pdf1 = stats.gaussian_kde(g1)
# pdf2 = stats.gaussian_kde(g2)
# pdf3 = stats.gaussian_kde(g3)
# pdf4 = stats.gaussian_kde(g4)

pdf1 = norm1
pdf2 = norm2
pdf3 = norm3
pdf4 = norm4


#plt.subplot(211)

plt.plot(x1,norm1(x1),label=r'$P_1$')
plt.plot(x2,norm2(x2),label=r'$P_2$')
plt.plot(x3,norm3(x3),label=r'$P_3$')
plt.plot(x4,norm4(x4),label=r'$P_4$')

plt.legend()
plt.grid(1)
plt.xlabel('x')
plt.ylabel(r'$P_i(x)$')

plt.show()

from renyi import *

## ENTROPIJA

# entropy1 = []
# shannon1 = []
# entropy2 = []
# entropy3 = []
# entropy4 = []
# shannon2 = []

# #a = renyi_entropy_cont(pdf1,1, minimum=min(g1),maximum=max(g1))
# #b = (renyi_entropy_cont(pdf2,1, minimum=min(g2),maximum=max(g2)))
# for i in np.arange(0.1, 3, 0.1):
#     entropy1.append(renyi_entropy_cont(pdf1,i, minimum=min(g1),maximum=max(g1)))
#     #shannon1.append(a)
#     entropy2.append(renyi_entropy_cont(pdf2,i, minimum=min(g2),maximum=max(g2)))
#     entropy3.append(renyi_entropy_cont(pdf3,i, minimum=min(g3),maximum=max(g3)))
#     entropy4.append(renyi_entropy_cont(pdf4,i, minimum=min(g4),maximum=max(g4)))
#     #shannon2.append(b)

# rang = np.arange(0.1, 3, 0.1)


# plt.plot(rang, entropy1,label=r'$H_\alpha(P_1)$')
# # plt.plot(rang, shannon1)
# plt.plot(rang, entropy2,label=r'$H_\alpha(P_2)$')
# plt.plot(rang, entropy3,label=r'$H_\alpha(P_3)$')
# plt.plot(rang, entropy4,label=r'$H_\alpha(P_4)$')
# # plt.plot(rang, shannon2)
# plt.legend()
# plt.grid(1)
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$H_\alpha(P_i)$')
# plt.show()

## DIVERGENCA

renyi12 = []
renyi13 = []
renyi14 = []
renyi21 = []
renyi23 = []
renyi24 = []
renyi31 = []
renyi32 = []
renyi34 = []
renyi41 = []
renyi42 = []
renyi43 = []


rang = np.arange(0.1,3,0.1)

for i in rang:
    renyi12.append(renyi_divergence_cont(pdf1,pdf2,i, -float('Inf'), float('Inf')))
    renyi13.append(renyi_divergence_cont(pdf1,pdf3,i, -float('Inf'), float('Inf')))
    renyi14.append(renyi_divergence_cont(pdf1,pdf4,i, -float('Inf'), float('Inf')))
    renyi21.append(renyi_divergence_cont(pdf2,pdf1,i, -float('Inf'), float('Inf')))
    renyi23.append(renyi_divergence_cont(pdf2,pdf3,i, -float('Inf'), float('Inf')))
    renyi24.append(renyi_divergence_cont(pdf2,pdf4,i, -float('Inf'), float('Inf')))
    renyi31.append(renyi_divergence_cont(pdf3,pdf1,i, -float('Inf'), float('Inf')))
    renyi32.append(renyi_divergence_cont(pdf3,pdf2,i, -float('Inf'), float('Inf')))
    renyi34.append(renyi_divergence_cont(pdf3,pdf4,i, -float('Inf'), float('Inf')))
    renyi41.append(renyi_divergence_cont(pdf4,pdf1,i, -float('Inf'), float('Inf')))
    renyi42.append(renyi_divergence_cont(pdf4,pdf2,i, -float('Inf'), float('Inf')))
    renyi43.append(renyi_divergence_cont(pdf4,pdf3,i, -float('Inf'), float('Inf')))

plt.subplot(121)
plt.plot(rang, renyi12, '.', label=r'$D_\alpha(P_1 || P_2)$',linewidth=4)
plt.plot(rang, renyi21, label=r'$D_\alpha(P_2 || P_1)$')
plt.plot(rang, renyi31, label=r'$D_\alpha(P_3 || P_1)$')
plt.plot(rang, renyi32, label=r'$D_\alpha(P_3 || P_2)$')
plt.plot(rang, renyi41, label=r'$D_\alpha(P_4 || P_1)$')
plt.plot(rang, renyi42, label=r'$D_\alpha(P_4 || P_2)$')
plt.plot(rang, renyi43, label=r'$D_\alpha(P_4 || P_3)$')

plt.legend()
plt.grid(1)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$D_\alpha(P_i || P_j)$')

plt.subplot(122)
plt.plot(rang, renyi13, label=r'$D_\alpha(P_1 || P_3)$')
plt.plot(rang, renyi14, label=r'$D_\alpha(P_1 || P_4)$')
plt.plot(rang, renyi23, label=r'$D_\alpha(P_2 || P_3)$')
plt.plot(rang, renyi24, label=r'$D_\alpha(P_2 || P_4)$')
plt.plot(rang, renyi34, label=r'$D_\alpha(P_3 || P_4)$')

plt.legend()
plt.grid(1)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$D_\alpha(P_i || P_j)$')
plt.subplots_adjust(wspace=0.2)
plt.show()