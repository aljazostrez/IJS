import random as r
import numpy as np


for i in range(0,100000000,1000000):
    if i==0:
        continue
    print(i)
    a = []
    for j in range(i):
        a.append(r.uniform(0,10))
    h, bins = np.histogram(a, bins=1000, density=1)
    min = 1000000000
    max = 0
    for i in range(len(h)-1):
        abso = abs(h[i]-h[i+1])
        if h[i] < min:
            min = h[i]
        if h[i] > max:
            max =h[i]
    try:
        if max/min - 1 < 0.1:
            print(max/min)
            break
    except:
        pass



    
# print(min)
# print(max)
# print("rel.: " + str(max/min))
# print("\n")