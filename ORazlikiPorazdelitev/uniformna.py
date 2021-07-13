import numpy as np
import matplotlib.pyplot as plt
import random as r
from math import pi, e

# x1 = np.arange(0,2.1,0.1)
# y1 = [0 for el in x1]
# x2 = np.arange(2,4.1,0.1)
# y2 = [0.5 for el in x2]
# x3 = np.arange(4,6.1,0.1)
# y3 = [0 for el in x3]

# plt.plot(x1,y1,color='blue')
# plt.plot(x2,y2,color='blue')
# plt.plot(x3,y3,color='blue')

# plt.axvline(x=2,color='blue', linestyle='--', ymin=0.05, ymax=0.5)
# plt.axvline(x=4,color='blue', linestyle='--', ymin=0.05, ymax=0.5)

# plt.plot([1,2,3], [1,1,1], color = 'white')

# plt.show()

def norm(x, mu, sigma):
        return (1/((2*pi*sigma**2)**0.5)) * e ** (-(x-mu)**2/(2*sigma**2))

def mix(x):
    return (1/3*(norm(x,-7,1)+norm(x,0,2)+norm(x,10,3)))

x = np.linspace(-15,20,50000)

plt.plot(x,mix(x))

plt.show()
