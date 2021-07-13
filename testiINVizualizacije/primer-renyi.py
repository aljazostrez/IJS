from renyi import *
import random
from scipy import stats, integrate
import numpy as np
import matplotlib.pyplot as plt


def sirina(sigma, N):
    return 3.49*(sigma)*(N**(-1/3))


with open('samples.txt') as dat:
    g1 = dat.readline().split(',')
    g2 = dat.readline().split(',')
    sample1 = list(map(float, g1))
    sample2 = list(map(float, g2))

x1 = np.linspace(min(sample1) -1, max(sample1)+1, len(sample1))
x2 = np.linspace(min(sample1) -1, max(sample1)+1, len(sample2))
h1 = np.arange(min(sample1), max(sample1), sirina(np.std(sample1), len(sample1)))
h2 = np.arange(min(sample2), max(sample2), sirina(np.std(sample2), len(sample2)))

plt.hist(sample2, bins=h1, density=1, edgecolor='black', linewidth=0.05)
plt.xlabel('x')
plt.grid(1)

plt.show()

pdf1 = stats.gaussian_kde(sample1)
pdf2 = stats.gaussian_kde(sample2)

ent1=[]
ent2=[]
div=[]

ran = np.arange(0, 5, 0.1)
minimum = min(min(sample1), min(sample2))
maximum = max(max(sample1), max(sample2))

for i in ran:
    ent1.append(renyi_entropy_cont(pdf1, i, minimum, maximum))
    ent2.append(renyi_entropy_cont(pdf2, i, minimum, maximum))
##    div.append(renyi_divergence_cont(pdf1,pdf2,i,minimum,maximum))
    

plt.plot(x1, pdf1(x1), label='p(x)')
plt.plot(x2, pdf2(x2), label='q(x)')
##plt.plot(ran, ent1, label='Entropija porazdelitve P')
##plt.plot(ran, ent2, label='Entropija porazdelitve Q')
##plt.plot(ran, div)
plt.legend()
plt.xlabel('x')
##plt.xlabel('x')
plt.ylabel('pdf(x)')
plt.grid(1)
plt.show()

#### ENTROPIJA IN DIVERGENCA GLEDE NA HISTOGRAM

def renyi_hist_entropy(x_array, y_array, alpha):
    area = 0
    if alpha == 1:
        for i in range(len(y_array)):
            if y_array[i] >= np.finfo(float).eps:
                area += y_array[i]*log(y_array[i], e)*(x_array[i+1]-x_array[i])
            else:
                pass
        return -area
    else:
        for i in range(len(y_array)):
            if y_array[i] >= np.finfo(float).eps:
                area += y_array[i]**alpha * (x_array[i+1]-x_array[i])
            else:
                pass
        return 1/(1-alpha)*log(area, e)

def renyi_hist_divergence(y1_array, y2_array, bins, alpha):
    y = []
    if alpha == 1:
        for i in range(len(y1_array)):
            if y1_array[i] >= np.finfo(float).eps and y2_array[i] >= np.finfo(float).eps:
                y.append(y1_array[i]*log(y1_array[i]/y2_array[i], e))
            elif y1_array[i] < np.finfo(float).eps:
                y.append(0.)
            else:
                y.append(float('Inf'))
    else:
        for i in range(len(y1_array)):
            if y1_array[i] >= np.finfo(float).eps and y2_array[i] >= np.finfo(float).eps:
                y.append(y1_array[i]**alpha * y2_array[i]**(1-alpha))
            elif y1_array[i] < np.finfo(float).eps:
                y.append(0.)
            else:
                y.append(float('Inf'))
    area = 0    
    for i in range(len(y)):
        area +=  y[i] * (bins[i+1]-bins[i])
    if alpha == 1:
        return area
    else:
        return 1/(alpha-1)*log(area, e)




s1 = []
s2 = []
minimum = min(min(sample1), min(sample2))
maximum = max(max(sample1), max(sample2))

for i in sample1:
    if minimum <= i <= maximum:
        s1.append(i)

for j in sample2:
    if minimum <= j <= maximum:
        s2.append(j)

n = 27
h = np.linspace(minimum, maximum, n+1)

y1, x1 = np.histogram(s1, bins=h, density=1)
y2, x2 = np.histogram(s2, bins=h, density=1)

ent=[]
ent1=[]
div = []
div1 = []

test = [6, 7, 8, 9, 10, 11, 12, 14]

#### ZA ENTROPIJO

for j in test:
    ent=[]
    for i in np.arange(0.1, 5, 0.1):
        n = j
        h = np.linspace(minimum, maximum, n+1)
        y1, x1 = np.histogram(s1, bins=h, density=1)
        y2, x2 = np.histogram(s2, bins=h, density=1)
        ent.append(renyi_hist_entropy(x2,y2,i))
    plt.plot(np.arange(0.1, 5, 0.1), ent, label='n={}'.format(str(n)))

for i in np.arange(0.1, 5, 0.1):
    ent1.append(renyi_entropy_cont(pdf2, i, minimum, maximum))

#### ZA DIVERGENCO
##for j in test:
##    div=[]
##    for i in np.arange(0.1, 5, 0.1):
##        n = j
##        h = np.linspace(minimum, maximum, n+1)
##        y1, x1 = np.histogram(s1, bins=h, density=1)
##        y2, x2 = np.histogram(s2, bins=h, density=1)
##        div.append(renyi_hist_divergence(y1,y2,h,i))
##    plt.plot(np.arange(0.1, 5, 0.1), div, label='n={}'.format(str(n)))

##for i in np.arange(0.1, 5, 0.1):
##    div1.append(renyi_divergence_cont(pdf1, pdf2, i, minimum, maximum))


zg = []
sp = []
for i in ent1:
    zg.append(i + 0.05)
    sp.append(i-0.05)

plt.plot(np.arange(0.1,5,0.1), ent1, label='po pdf', color='black', linewidth=2)
plt.plot(np.arange(0.1,5,0.1), zg, color='black', linestyle='--', linewidth=2, label='pdf +/- 0.05')
plt.plot(np.arange(0.1,5,0.1), sp, color='black', linestyle='--', linewidth=2)

plt.grid(1)
plt.xlabel('α')
plt.ylabel('Entropija')
plt.legend()
plt.show()


