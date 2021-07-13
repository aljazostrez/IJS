import random as r
from scipy import stats
import numpy as np
from renyi import JRD_pdfs
import matplotlib.pyplot as plt
import csv

r.seed(0)
koliko = 18

normalne_z_repi  = {}


for i in range(koliko):
    a = []
    for j in range(100000):
        if i == 0:
            a.append(r.gauss(0,1))
        elif i < 9:
            if j < 80000:
                a.append(r.gauss(0,1))
            elif 80000 <= j < 90000:
                a.append(r.gauss(i*3,1))
            else:
                a.append(r.gauss(-i*3,1))
        else:
            if j < 80000:
                a.append(r.gauss((i-9)*6,1))
            elif 80000 <= j < 90000:
                a.append(r.gauss((i-9)*6+18,1))
            else:
                a.append(r.gauss((i-9)*6-18,1))
    normalne_z_repi[i] = a


pdfs = {}
for i in range(koliko):
    pdfs[i] = stats.gaussian_kde(normalne_z_repi[i])

def linspace(sample):
    return np.linspace(min(normalne_z_repi[8]), max(normalne_z_repi[17]), 10000)

for i in range(0,18):
    if i == 0:
        plt.figure("Normalne z repi 1")
        ax = plt.subplot(3,3,1)
    if i == 9:
        plt.figure("Normalne z repi 2")
    if 0 < i < 9:
        plt.subplot(3,3,i+1, sharex=ax, sharey=ax)
    elif 9 <= i:
        plt.subplot(3,3,i-9+1, sharex=ax, sharey=ax)
    plt.plot(linspace(8), pdfs[i](linspace(17)))
    plt.title("pdf {}".format(i+1))
    plt.grid(1)
    print(i)

plt.show()

JRDs = []

for i in range(koliko):
    JRD = []
    JRD.append("pdf 1 in pdf {}".format(i+1))
    for j in np.arange(0,3,0.2):
        print(i,j)
        try:
            JRD.append(round(JRD_pdfs([pdfs[0],pdfs[i]],j,minimum=min(normalne_z_repi[0]+normalne_z_repi[i]), maximum=max(normalne_z_repi[0]+normalne_z_repi[i])),4))
        except:
            JRD.append(float('inf'))
    JRDs.append(JRD)

h = [" "] + ["α={}".format(round(i,2)) for i in np.arange(0,3,0.2)]


ent = [h] + JRDs

with open("Aljaz/JRDNormalnihZRepi.csv", 'w', encoding='utf-8') as csv_datoteka:
    writer = csv.writer(csv_datoteka)
    for el in ent:
        writer.writerow(el)


# plt.show()

    