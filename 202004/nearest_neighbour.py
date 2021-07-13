from math import sqrt
import numpy as np

def nearest_neighbour(data,k=None):
    n = len(data)
    data = np.array(data)
    if k==None:
        k = int(sqrt(n))
        k = int(n/2)

    def f(x):
        distances = np.abs(data-x)
        distances.sort()
        
        return k/(2*n*distances[k-1])

    return f



# ## TEST
# import random as r
# import numpy as np
# import matplotlib.pyplot as plt
# from GMM import norm

# r.seed(0)

# dat = []
# for i in range(100000):
#     dat.append(r.gauss(0,1))
# print("generated")

# pdf = nearest_neighbour(dat)

# x = np.linspace(-3,3,1000)
# pdf_izris = [pdf(i) for i in x]


# plt.plot(x, pdf_izris)
# plt.plot(x,norm(x,0,1),label="prava")
# plt.legend()
# plt.show()
