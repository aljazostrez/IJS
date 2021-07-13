from Read_data import *
import matplotlib.pyplot as plt


celica = 5
meritev = 28


# marginalni_histogrami(celica, meritev)

## PREVERJANJE, ČE SO HISTOGRAMI NORMALIZIRANI: NISO!
def vsota(marg, celica, meritev):
    w = cell[celica][marg+"_bin"][meritev][1] - cell[celica][marg+"_bin"][meritev][0]
    sum = 0
    for i in range(30):
        sum += w*cell[celica][marg+"_pdf"][meritev][i]
    return sum

def preveri_indekse(marg, celica):
    sum = 0    
    for i in range(511):
        if abs(vsota(marg,celica,i) -1) > 0.1:
            sum += 1
            # print(i)
    # print("\n")
    # print("\n")
    return "stevilo vseh: " + str(sum)

params = ['Rs','R1','R2','R3','tau1','tau2','tau3','alfa1','alfa2','alfa3','sigma']

def preveri():
    for i in range(6):
        for par in params:
            print("celica: {}, parameter: {}, {}".format(str(i), par, preveri_indekse(par,i)))

# for j  in range(6):
#     for i in range(0,501,100):
#         marginalni_histogrami(j,i)

## DOBIMO IZ POZICIJE CENTROV POZICIJE MEJ
def center_to_edges(ar):
    min = float("inf")
    max = -1
    for i in range(len(ar)-1):
        w = ar[i+1]-ar[i]
        if w > max:
            max = w
        if w < min:
            min = w
    # if abs(max/min - 1) > 0.01:
    #     return("Sirine stolpcev niso konstantne")
    bins = []
    for el in ar:
        bins.append(el-w/2)
    bins.append(ar[-1]+w/2)
    return np.array(bins)


## PRVI POSKUS ZA RAČUNANJE DIVERGENCE
# params1 = ['R1','R2','R3','tau1','tau2','tau3','alfa1','alfa2','alfa3']

# cell[0]["tau1_bin"] = cell[0]["tau1_bin"]
# cell[0]["tau1_pdf"] = cell[0]["tau1_pdf"]

# x1 = center_to_edges(cell[0]["tau1_bin"][20])
# y1 = cell[0]["tau1_pdf"][20]

# renyi_t = []
# for i in range(512):
#     print(i)
#     sum = 0
#     for j in range(len(cell)):
#         for parameter in params1:
#             x1 = center_to_edges(cell[j]["{}_bin".format(parameter)][20])
#             y1 = cell[j]["{}_pdf".format(parameter)][20]
#             x2 = center_to_edges(cell[j]["{}_bin".format(parameter)][i])
#             y2 = cell[j]["{}_pdf".format(parameter)][i]
#             sum += renyi_divergence_hist(x1,y1,x2,y2,5)
#     renyi_t.append(sum)


# plt.figure()
# plt.plot(range(512), renyi_t)
# plt.grid(1)

# plt.figure()
# a = plt.subplot(211)
# w = cell[0]["tau1_bin"][20][1]-cell[0]["tau1_bin"][20][0]
# plt.bar(x=cell[0]["tau1_bin"][20], height=cell[0]["tau1_pdf"][20], width=w)

# plt.subplot(212, sharex=a)
# second = 10
# w = cell[0]["tau1_bin"][second][1]-cell[0]["tau1_bin"][second][0]
# plt.bar(x=cell[0]["tau1_bin"][second], height=cell[0]["tau1_pdf"][second], width=w)

# plt.show()


## NORMALIZACIJA HISTOGRAMOV

count = 0
for i in range(len(cell)):
    for par in params:
        for j in range(len(cell[i][par+"_pdf"])):
            p = vsota(par,i,j)
            if round(p,5) != 1 and p!=0:
                for k in range(len(cell[i][par+"_pdf"][j])):
                    # cell[st.celice][parameter][]
                    cell[i][par+"_pdf"][j][k] = cell[i][par+"_pdf"][j][k]/p
                    count += 1
                # print("Urejen: celica {}, parameter {}, meritev {}".format(str(i), par, str(j)))

print("Normalizacija histogramov končana.")

def histogram_par(par, celica, meritev):
    return (cell[celica][par+"_pdf"][meritev], center_to_edges(cell[celica][par+"_bin"][meritev]))

histogrami = {}
for i in range(len(cell)):
    histogrami[i] = {}
    for par in params:
        histogrami[i][par] = {}
        for j in range(len(cell[i][par+"_pdf"])):
            histogrami[i][par][j] = histogram_par(par,i,j)


def JRD_par(parameter, celica, second_index, fiksna=20):
    hist1 = histogrami[celica][parameter][fiksna]
    hist2 = histogrami[celica][parameter][second_index]
    return generalised_RD([hist1,hist2], 0.5)

def JRD_all(celica, second_index, red, fiksna=20, Rs=0, sigma=0, weight=None):
    params = ['R1','R2','R3','tau1','tau2','tau3','alfa1','alfa2','alfa3']
    if Rs:
        params.append('Rs')
    if sigma:
        params.append('sigma')
    hists1 = []
    hists2 = []
    for parameter in params:
        hists1.append(histogrami[celica][parameter][fiksna])
        hists2.append(histogrami[celica][parameter][second_index])
    value = generalised_RD(hists1, red, weigths=weight) - generalised_RD(hists2, red, weigths=weight)
    return value
        

import time



error_index = list(range(5))+list(range(44,58))+list(range(96,112))+[238]+[313,314]
print(error_index)

## MODEL STEVILKA 1
def JRD(w=None):    
    JRD = []
    for i in range(512):
        sum = 0
        for j in range(6):
            # print(i,j)
            sum += JRD_all(j,i,3,weight=w)
        JRD.append((abs(sum),i))
    return JRD

## MODEL STEVILKA 2
def JRD2(w):
    params1 = ['R1','R2','R3','tau1','tau2','tau3','alfa1','alfa2','alfa3']
    renyi_t = []
    for i in range(512):
        sum = 0
        for j in range(len(cell)):
            for k in range(len(params1)):
                x1 = center_to_edges(cell[j]["{}_bin".format(params1[k])][20])
                y1 = cell[j]["{}_pdf".format(params1[k])][20]
                x2 = center_to_edges(cell[j]["{}_bin".format(params1[k])][i])
                y2 = cell[j]["{}_pdf".format(params1[k])][i]
                sum += w[k] * renyi_divergence_hist(x1,y1,x2,y2,3)
        renyi_t.append((sum,i))
    return renyi_t
    

import sys, os

# Disable
def blockPrint():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return old_stdout

# Restore
def enablePrint(a):
    sys.stdout = a


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from skopt import gp_minimize
from skopt.plots import plot_convergence





def f(x):
    # if(round(sum(x),3)!=1):
    #     return 10000
    printing=blockPrint()
    res=sorted(JRD2(x),reverse=True)
    enablePrint(printing)
    ind=[]
    for i in range(512):
        ind.append(res[i][1])
        if(set(error_index)<=set(ind)):
            break
    return len(ind)

score_269 = [0.0, 41.387384457920554, 0.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0]

# res = gp_minimize(f,                  # the function to minimize
#                   [(0.0, 100), (0.0,100),(0.0, 100), (0.0,100),(0.0, 100), (0.0,100),(0.0, 100), (0.0,100),(0.0, 100)],      # the bounds on each dimension of x
#                 #   acq_func="EI",      # the acquisition function
#                   n_calls=1000,         # the number of evaluations of f
#                   x0=score_269,
#                   n_random_starts=5,  # the number of random initialization points
#                   verbose=True)
#                   #noise=0.1**2,       # the noise level (optional)
#                   #random_state=101)   # the random seed

# print((res.x, res.fun))

# import random

# def random_list():
#     l = []
#     l.append(random.uniform(0,1))
#     for i in range(7):
#         l.append(random.uniform(0,1-sum(l)))
#     l.append(1-sum(l))
#     random.shuffle(l)
#     return l

# def random_list1():
#     l = []
#     for i in range(9):
#         l.append(random.randint(0,100))
#     return l

# length = 292
# ugodne_utezi = [100.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0]
# for i in range(100):
#     print(i)
#     w = random_list1()
#     l = f(w)
#     print(w,l)
#     if l < length:
#         ugodne_utezi = w
#         length = l
#         print(l)

# W = []
# for a in random.sample(range(101), 10):
#     for b in random.sample(range(101), 10):
#         for c in random.sample(range(101), 10):
#             for d in random.sample(range(101), 10):
#                 for e in random.sample(range(101), 10):
#                     for z in random.sample(range(101), 10):
#                         for g in random.sample(range(101), 10):
#                             for h in random.sample(range(101), 10):
#                                 for i in random.sample(range(101), 10):
#                                     w = [a,b,c,d,e,z,g,h,i]
#                                     W.append(w)
                                    
# random.shuffle(W)
# print("W koncan")

# for weight in W:
#     l = f(weight)
#     print(weight,l)
#     if l < length:
#         ugodne_utezi = weight
#         length = l
#         print(l)


# count = 0 
# zac = time.time()
# for a in np.arange(0,1.01,0.1):
#     if a<0: continue
#     for b in np.arange(0, 1.01-a, 0.1):
#         if b<0: continue
#         for c in np.arange(0, 1.01-a-b,0.1):
#             if c<0: continue
#             for d in np.arange(0, 1.01-a-b-c,0.1):
#                 if d<0: continue
#                 for e in np.arange(0, 1.01-a-b-c-d,0.1):
#                     if e<0: continue
#                     for f in np.arange(0, 1.01-a-b-c-d-e,0.1):
#                         if f<0: continue
#                         for g in np.arange(0, 1.01-a-b-c-d-e-f,0.1):
#                             if g<0: continue
#                             for h in np.arange(0, 1.01-a-b-c-d-e-f-g,0.1):
#                                 if h<0: continue
#                                 for z in np.arange(0, 1.01-a-b-c-d-e-f-g-h,0.1):
#                                     if z<0: continue
#                                     w = [a,b,c,d,e,f,g,h,z]
#                                     if sum(w) == 1:
#                                         print(count)
#                                         printing = blockPrint()
#                                         res = sorted(JRD(w=w), reverse=True)
#                                         enablePrint(printing)
#                                         ind = [res[i][1] for i in range(511)]
#                                         if set(error_index) <= set(ind):
#                                             print(w, ind)
#                                         count += 1
# kon = time.time() - zac
# print("Trajalo je {} sekund.".format(str(kon)))



# plt.plot(range(512), JRD)
# plt.grid(1)
# plt.show()