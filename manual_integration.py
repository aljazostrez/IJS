import numpy as np
from time import time
from scipy import integrate

def integrate_num(f, a, b):
    # zac = time()
    if b-a<0:
        return("b must be bigger than a")
    elif b-a>1:
        x = np.arange(a,b,0.01)
    else:
        x = np.linspace(a,b,100)
    w = x[1]-x[0]
    area = 0
    for xi in x[:-1]:
        area += f(xi)*w
    # kon = time()
    # print("Cas: " + str(kon-zac))
    return area

# def f(x): return x**2

# print(integrate_num(f,-5,5))

# print("\n\n")

# zac = time()
# print(integrate.quad(f,-5,5))
# kon=time()
# print("Cas: " + str(kon-zac) + "\n")