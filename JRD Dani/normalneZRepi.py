import random as r
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from renyi import renyi_entropy_cont, renyi_hist_entropy
# from tabulate import tabulate
import csv


r.seed(0)
koliko = 9

normalne_z_repi = {}

for i in range(koliko):
    a = []
    for j in range(100000):
        if j < 80000:
            a.append(r.gauss(0,1))
        elif 80000 <= j < 90000:
            a.append(r.gauss(i*3,1))
        else:
            a.append(r.gauss(-i*3,1))
    normalne_z_repi[i] = a

hists = {}
for i in range(koliko):
    hists[i] = np.histogram(normalne_z_repi[i],bins=15, density=1)

pdfs = {}
for i in range(koliko):
    pdfs[i] = stats.gaussian_kde(normalne_z_repi[i])

entropije = []

# for i in range(koliko):
#     entropija = []
#     entropija.append("pdf {}".format(i+1))
#     for j in np.arange(0,3,0.2):
#         print(i,j)
#         if j == 1:
#             entropija.append(round(renyi_entropy_cont(pdfs[i],j,minimum=min(normalne_z_repi[i]),maximum=max(normalne_z_repi[i])),4))
#         else:
#             entropija.append(round(renyi_entropy_cont(pdfs[i],j,minimum=min(normalne_z_repi[i]),maximum=max(normalne_z_repi[i])),4))
#     entropije.append(entropija)




# def linspace(sample):
#     return np.linspace(min(normalne_z_repi[sample]), max(normalne_z_repi[sample]), 10000)

# for i in range(koliko):
#     if i == 0:
#         ax = plt.subplot(3,3,1)
#     else:
#         plt.subplot(3,3,i+1, sharex=ax, sharey=ax)
#     plt.title("pdf {}".format(i+1))
#     plt.plot(linspace(8), pdfs[i](linspace(8)))
#     plt.grid(1)
#     print(i)


h = ["no.\α"] + ["α={}".format(round(i,2)) for i in np.arange(0,3,0.2)]

# print(tabulate(entropije,headers=h, tablefmt='orgtbl'))


ent = [h] + entropije

# with open("Aljaz/entropijeNormalnihZRepi.csv", 'w', encoding='utf-8') as csv_datoteka:
#     writer = csv.writer(csv_datoteka)
#     for el in ent:
#         writer.writerow(el)



# plt.plot(range(koliko), entropije)
# plt.legend()

plt.show()