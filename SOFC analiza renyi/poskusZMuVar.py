from Read_data import *
from math import log, e

params = ['Rs','R1','R2','R3','tau1','tau2','tau3','alfa1','alfa2','alfa3','sigma']

def renyi_mu_var(mu_1, sigma_1, mu_2, sigma_2, alpha):
    if alpha != 1:
        var_zvezdica = alpha*(sigma_1 ** 2) + (1-alpha)*(sigma_2**2)
        try:
            rezultat = log(sigma_1/sigma_2, e) + (1/(2*(alpha-1)))*log((sigma_1**2)/var_zvezdica, e) + (alpha*((mu_2 - mu_1)**2))/(2*var_zvezdica)
            return rezultat
        except ValueError:
            return 0
    else:
        try:
            rezultat = (1/(2*(sigma_1**2)))*((mu_2 - mu_1)**2 + sigma_2**2 - sigma_1**2) + log(sigma_1/sigma_2, e)
            return rezultat
        except ValueError:
            return 0

def mu_var_pair(parameter, celica, meritev):
    return (cell[celica][par+"_mu"][meritev],cell[celica][par+"_var"][meritev])

mu_var = {}

for i in range(len(cell)):
    mu_var[i] = {}
    for par in params:
        mu_var[i][par] = {}
        for j in range(len(cell[i][par+"_mu"])):
            mu_var[i][par][j] = mu_var_pair(par,i,j)

def RD_mu_var(parameter, celica, second_index, fiksna=20):
    mu1 = mu_var[celica][parameter][fiksna][0][0]
    sigma1 = mu_var[celica][parameter][fiksna][1][0]
    mu2 = mu_var[celica][parameter][second_index][0][0]
    sigma2 = mu_var[celica][parameter][second_index][1][0]
    return renyi_mu_var(mu1,sigma1,mu2,sigma2,2)

w = [0.16588309513330696, 9.358882117868609, 0.19449510141646711, 6.688613298883591, 9.44049131831504, 6.889740145741952, 7.228298300590705, 5.164063630903483, 2.4470834311477203]

def RD_final(w):
    params1 = ['R1','R2','R3','tau1','tau2','tau3','alfa1','alfa2','alfa3']
    renyi_final = []
    for i in range(512):
        sum = 0
        for j in range(len(cell)):
            for k in range(len(params1)):
                sum += w[k] * RD_mu_var(params1[k],j,i)
        renyi_final.append((sum,i))
    return renyi_final


from mpl_toolkits.mplot3d import Axes3D
from skopt import gp_minimize
from skopt.plots import plot_convergence

error_index = list(range(5))+list(range(44,58))+list(range(96,112))+[238]+[313,314]

def f(x):
    # if(round(sum(x),3)!=1):
    #     return 10000
    res=sorted(RD_final(x),reverse=True)
    ind=[]
    for i in range(512):
        ind.append(res[i][1])
        if(set(error_index)<=set(ind)):
            break
    return len(ind)


# res = gp_minimize(f,                  # the function to minimize
#                   [(0.0, 10), (0.0,10),(0.0, 10), (0.0,10),(0.0, 10), (0.0,10),(0.0, 10), (0.0,10),(0.0, 10)],      # the bounds on each dimension of x
#                 #   acq_func="EI",      # the acquisition function
#                   n_calls=200,         # the number of evaluations of f
#                 #   x0=score_269,
#                   n_random_starts=5,  # the number of random initialization points
#                   verbose=1)
#                   #noise=0.1**2,       # the noise level (optional)
#                   #random_state=101)   # the random seed

# print((res.x, res.fun))