import numpy as np
import matplotlib.pyplot as plt
from math import pi,e

def norm(x, mu, sigma):
    return (1/((2*pi*sigma**2)**0.5)) * e ** (-(x-mu)**2/(2*sigma**2))
        
def n(x,y):
    return norm(x,y,10)

x = np.linspace(-40,40,1000)
for i in range(10):
    plt.plot(x, n(x,i))

plt.grid(1)
plt.show()
