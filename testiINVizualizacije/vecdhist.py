import matplotlib.pyplot as plt
import random as r
import numpy as np

r.seed(0)
x = []
y = []
z = []
u = []

for i in range(10000):
    x.append(r.gauss(0,20))
    y.append(3*r.gauss(0,20)+r.gauss(10,5))
    z.append(r.gauss(30, 5))
    u.append(r.gauss(30,1))

# fig, ax = plt.subplots()
# h = ax.hist2d(x,y,bins=20, normed=0)
# plt.colorbar(h[3], ax=ax)
# plt.show()


# heigths, x1, y1 = np.histogram2d(x,y)


h, bins = np.histogramdd((x,y), bins=5, normed=1)
nphist = np.histogramdd((x,y), bins=(6,5), normed=1)
h1, bins1 = np.histogramdd((z,u), bins=5,normed=1)

ar = []
for i in range(len(h)):
    ar1 = []
    loop = len(h[i])
    for j in range(loop):
        ar1.append((bins[0][i], bins[1][j], h[i][j]))
    ar1.append((bins[0][i], bins[1][loop], None))
    ar.append(ar1)

ar1 = []
for i in range(len(bins[1])):
    m = len(bins[0])-1
    ar1.append((bins[0][m], bins[1][i], None))
ar.append(ar1)


# # VOLUMEN JE RES ENAK 1
# sum = 0
# for i in range(len(ar)):
#     for j in range(len(ar[i])):
#         if ar[i][j][2] != None:
#             s = (ar[i+1][j][0] - ar[i][j][0]) * (ar[i][j+1][1] - ar[i][j][1]) * ar[i][j][2]
#             sum += s

# a = np.array(ar)


# from renyi import renyi_divergence_2d, renyi_entropy_2d
# plt.subplot(211)
# plt.hist2d(x,y)
# plt.subplot(212)
# plt.hist2d(z,u)
# plt.show()

# re2d1 = []
# re2d2 = []
# rd2d = []
# for i in np.arange(0.1,2,0.1):
#     rd2d.append(renyi_divergence_2d(bins, h, bins1, h, i))
#     re2d1.append(renyi_entropy_2d(bins,h,i))
#     re2d2.append(renyi_entropy_2d(bins1,h1,i))

# plt.subplot(211)
# plt.plot(np.arange(0.1,2,0.1), re2d1)
# plt.plot(np.arange(0.1,2,0.1), re2d2)
# plt.grid(1)

# plt.subplot(212)
# plt.plot(np.arange(0.1,2,0.1), rd2d)
# plt.grid(1)

# plt.show()