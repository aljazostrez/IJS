from renyi import *
from scipy import stats, integrate
import numpy as np
from math import log, e
import matplotlib.pyplot as plt
import ast
import time

start = time.time()

with open('gauss.txt') as dat:
	gauss = ast.literal_eval(dat.readline())


pdf = {}
for i in range(10):
    pdf[i+1] = stats.gaussian_kde(gauss[i+1])


x = np.linspace(min(gauss[10]), max(gauss[10]), 1000)


subplot1 = plt.subplot(211)
##subplot1.plot(x, pdf[1](x), color='black', label='pdf1')
subplot2 = plt.subplot(212)

for i in range(1,11,1):
    renyi = []
    for j in np.arange(0.1,2,0.1):
        minimum = min([min(np.array(gauss[1])), min(np.array(gauss[i]))])
        maximum = max([max(np.array(gauss[1])), max(np.array(gauss[i]))])
        renyi.append(renyi_divergence_cont(pdf[1], pdf[i], j, minimum, maximum))
        print(i, ',', round(j,1), ': ', renyi_divergence_cont(pdf[1], pdf[i], j, -float('Inf'), float('Inf')))
    subplot1.plot(x, pdf[i](x), label='pdf{}'.format(str(i)))
    subplot2.plot(np.arange(0.1,2,0.1), renyi, label='pdf1_pdf{}'.format(str(i)))

subplot1.legend()
subplot1.grid(1)
subplot2.legend()
subplot2.grid(1)

end = time.time()
print(end-start, 'sec')

plt.show()
