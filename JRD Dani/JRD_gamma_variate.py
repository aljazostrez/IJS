import random as r
from scipy import stats
import numpy as np
from renyi import JRD_pdfs
import matplotlib.pyplot as plt
import csv

r.seed(0)
koliko = 9

gammav_porazdelitve = {}

for i in range(koliko):
    values = []
    for j in range(100000):
        values.append(r.gammavariate((i+1)/2,2))
    gammav_porazdelitve[i] = values


pdfs = {}
for i in range(koliko):
    pdfs[i] = stats.gaussian_kde(gammav_porazdelitve[i])

lin = np.linspace(0,20,2000)

# for i in range(koliko):
#     if i == 0:
#         ax = plt.subplot(3,3,1)
#     else:
#         plt.subplot(3,3,i+1, sharex=ax,sharey=ax)
#     plt.plot(lin, pdfs[i](lin))
#     plt.title("pdf {}, α={}, β={}".format(i+1,(i+1)/2,2))
#     plt.grid(1)
#     print(i)

JRDs = []

for i in range(koliko):
    JRD = []
    JRD.append("pdf 1 in pdf {}".format(i+1))
    for j in np.arange(0,3,0.2):
        print(i,j)
        try:
            JRD.append(round(JRD_pdfs([pdfs[0],pdfs[i]],j,minimum=-1, maximum=max(gammav_porazdelitve[8])),4))
        except:
            JRD.append(float('inf'))
    JRDs.append(JRD)

h = [" "] + ["α={}".format(round(i,2)) for i in np.arange(0,3,0.2)]

ent = [h] + JRDs

with open("Aljaz/JRDGammaDistributions.csv", 'w', encoding='utf-8') as csv_datoteka:
    writer = csv.writer(csv_datoteka)
    for el in ent:
        writer.writerow(el)

