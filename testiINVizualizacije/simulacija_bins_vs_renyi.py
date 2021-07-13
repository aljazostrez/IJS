# -*- coding: utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from renyi import *
from scipy import stats


## Za odpranje podatkov, shranjenih v datoteki twogausses.txt
with open('twogausses.txt') as dat:
    g1 = dat.readline().split(',')
    g2 = dat.readline().split(',')
    gauss1 = list(map(float, g1))
    gauss2 = list(map(float, g2))

    
## Da se ovrže vse podatke, ki niso v istem definicijskem obmocju
gauss = []
for x in gauss2:
    if x >= min(gauss1) and x <= max(gauss1):
        gauss.append(x)
gauss2 = gauss


## Da bosta imela oba histograma iste bine
h = np.linspace(min(gauss1), max(gauss1), 20 + 1)


## Oba histograma (ista širina binov, isti razpon)
y1, x1 = np.histogram(gauss1, bins=h, density=1)
y2, x2 = np.histogram(gauss2, bins=h, density=1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

a = 11

x = np.arange(0.1,3,0.1)
y = list(range(1,a))
x,y = np.meshgrid(x,y)


# ODKOMENTIRAJ VSE TO ZA ENTROPIJO
#z = []
#
#
#for i in range(1,a):
#    z.append([])
#    for j in np.arange(0.1,3,0.1):
#        h = np.linspace(min(gauss1), max(gauss1), i + 1)
#        y1, x1 = np.histogram(gauss1, bins=h, density=1)
#        y2, x2 = np.histogram(gauss2, bins=h, density=1)
#        z[i-1].append(renyi_hist_entropy(x1,y1,j))
#
#z = np.array(z)
#
#ax.set_xlabel('Alpha')
#ax.set_ylabel('Število stolpcev')
#ax.set_zlabel('Entropija')
#
#ax.set_xticks(np.arange(0.1,3,0.5))
#ax.set_yticks(range(1,a))
#
#
#line = ax.plot_surface(x,y,z,rstride=1, cstride=2, label='Ent. v odvisnosti od reda in št. stolpcev')
#ax.view_init(elev=18, azim=18)
#
#pdf = stats.gaussian_kde(gauss1)
#
#renyi = []
#for i in np.arange(0.1,3,0.1):
#    renyi.append(renyi_entropy_cont(pdf, i, min(gauss1), max(gauss1)))
#
#
#x = np.arange(0.1,3,0.1)
#y = list([a-1 for i in range(1,30)])
#
#ax.plot(x,y,renyi,color='red', label='Ent. glede na gostoto verjetnosti')
#
###ax.legend()
#
#plt.show()


z=[]
for i in range(1,a):
    z.append([])
    for j in np.arange(0.1,3,0.1):
        h = np.linspace(min(gauss1), max(gauss1), i + 1)
        y1, x1 = np.histogram(gauss1, bins=h, density=1)
        y2, x2 = np.histogram(gauss2, bins=h, density=1)
        z[i-1].append(renyi_divergence_hist(x1,y1,x2,y2,j))

z = np.array(z)

ax.set_xlabel('Alpha')
ax.set_ylabel('Število stolpcev')
ax.set_zlabel('Divergenca')

ax.set_xticks(np.arange(0.1,3,0.5))
ax.set_yticks(range(1,a))


line = ax.plot_surface(x,y,z,rstride=1, cstride=2, label='Div. v odvisnosti od reda in št. stolpcev')
ax.view_init(elev=22, azim=-153)

pdf = stats.gaussian_kde(gauss1)
pdf2 = stats.gaussian_kde(gauss2)

renyi = []
for i in np.arange(0.1,3,0.1):
    renyi.append(renyi_divergence_cont(pdf, pdf2, i, min(gauss1), max(gauss1)))


x = np.arange(0.1,3,0.1)
y = list([a-1 for i in range(1,30)])

ax.plot(x,y,renyi,color='red', label='Div. glede na gostoto verjetnosti')

##ax.legend()

plt.show()