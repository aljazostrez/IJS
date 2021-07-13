import random as r
import numpy as np
from scipy import stats
import ast
from renyi import *

with open("gaussJR.txt", 'r') as dat:
    line = dat.readline()
    g = ast.literal_eval(line)

def generelized_RD(histograms, alpha, weigths=None):
    if weigths == None:
        weigths = [1/len(histograms) for i in histograms]
    else:
        if len(weigths) != len(histograms):
            raise ValueError("Length of the weigths list must be the same"
            "as number of the histograms"
            )
        # Natancnost vsote utezi na 5 decimalk
        if round(sum(weigths[:len(histograms)]), 5) != 1:
            raise ValueError("Sum of the weigths is not equal to 1")
    hist1 = histograms[0]
    for i in range(1, len(histograms)):
        y1,x1 = hist1
        y2,x2 = histograms[i]
        a,b,c = uredi_stolpce(x1,y1,x2,y2, zeros=1)
        hist1 = (b,a)
    y1,x1 = hist1
    histograms1 = []
    for j in range(1, len(histograms)):
        y2,x2 = histograms[j]
        a,b,c = uredi_stolpce(x1,y1,x2,y2, zeros=1)
        histograms1.append((c,a))
    histograms1.append(hist1)
    all_y = [0 for i in histograms1[0][0]]
    for i in range(len(all_y)):
        for j in range(len(histograms1)):
            all_y[i] += weigths[j] * histograms1[j][0][i]
    x = hist1[1]
    sum2 = 0
    for i in range(len(histograms)):
        sum2 += weigths[i] * renyi_hist_entropy(histograms[i][1],histograms[i][0], alpha)
    return renyi_hist_entropy(x, all_y, alpha) - sum2

hists = [np.histogram(g[i], bins = 20, density=1) for i in range(len(g))]

import matplotlib.pyplot as plt

x_ax = np.linspace(-40,40,10000)

data = [g[i] for i in range(len(g))]

# pdfs = []
# for d in data:
#     pdfs.append(stats.gaussian_kde(d))
    
# for pdf in pdfs:
#     plt.plot(x_ax, pdf(x_ax))

entropy = []
rang = np.arange(0.1,5,0.1)
weigths = [0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

for i in rang:
    entropy.append(generelized_RD(hists,i, weigths=weigths))

plt.plot(rang, entropy)
plt.grid(1)
plt.show()