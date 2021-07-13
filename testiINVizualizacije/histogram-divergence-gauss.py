import numpy as np
import matplotlib.pyplot as plt
from math import log, e
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


## Entropija glede na histogram
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
## po preizkusu: najbolje je, ce je število binov od 10-15. Ce jih je prevec (tj. ima kateri od binov vrednost 0),
## je približek slabši. To se zgodi le ob zelo nelogicni izbiri stevila binov (npr. veliko prevec binov)


## Divergenca glede na histograma
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


#### Izracun pdf za primerjavo divergence po definiciji z divergenco glede na histogram
##pdf1 = stats.gaussian_kde(gauss1)
##pdf2 = stats.gaussian_kde(gauss2)
##minimum = min([min(gauss1), min(gauss2)])
##maximum = max([max(gauss1), max(gauss2)])
##
##
#### Primerjava
##entropija = []
##entropija1 = []
##
##renyi = []
##renyi1 = []
##
##print('Število binov je {}'.format(str(len(h)-1)))
##
##
##for i in np.arange(0.1, 2,0.1):
##    a = renyi_hist_entropy(h, y1, i)
##    b = renyi_entropy_cont(pdf1, i, min(x1), max(x1))
##    print('Entropija (alpha={})'.format(str(i)), '\n    glede na histogram: ', round(a,3), '\n    glede na pdf: ', round(b,3))
##    entropija.append(a)
##    entropija1.append(b)
##    
##    c = renyi_hist_divergence(y1, y2, h, i)
##    d = renyi_divergence_cont(pdf1, pdf2, i, minimum, maximum)
##    print('Divergenca (alpha={})'.format(str(i)), '\n    glede na histograma: ', round(c, 3), '\n    glede na pdfa: ', round(d,3))
##    renyi.append(c)
##    renyi1.append(d)
##
##
##subplot1 = plt.subplot(212)
##plt.plot(np.arange(0.1, 2, 0.1), renyi, color='red', label='po hist')
##plt.plot(np.arange(0.1, 2, 0.1), renyi1, color='blue', label='po pdf')
##plt.title('Divergenca v odvisnosti od alpha', fontsize=18)
##plt.xlabel('Alpha')
##plt.ylabel('Divergenca')
##plt.grid(1)
##plt.legend()
##
##subplot2 = plt.subplot(211)
##plt.plot(np.arange(0.1, 2, 0.1), entropija, color='red', label='po hist')
##plt.plot(np.arange(0.1, 2, 0.1), entropija1, color='blue', label='po pdf')
##plt.title('Entropija v odvisnosti od alpha', fontsize=18)
##plt.xlabel('Alpha')
##plt.ylabel('Entropija')
##plt.grid(1)
##plt.legend()
##
##plt.show()
##
