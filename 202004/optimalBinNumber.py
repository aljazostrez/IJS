import numpy as np
from math import sqrt, log, ceil
from scipy.stats import kurtosis, iqr, gaussian_kde
import scipy
from manual_integration import integrate_num
import matplotlib.pyplot as plt

def optBinNum(data, choice="scoot", n_z=10, n_k=100, izris=False):
    # data: set podatkov
    # choice: metoda za izracun:
    #     -> sqrt: n = ceil(sqrt(len(data)))
    #     -> scott: w = 3.49*std(data)*len(data)**(-1/3), n = ceil((max(data)-min(data))/w)
    #     -> rice: n = ceil(2*len(data)**(1/3))
    #     -> sturges: n = ceil(log(len(data), 2))+1
    #     -> fd: (Freedman-Diaconis) w = 2*iqr(data)*len(data)**(-1/3), n = ceil((max(data)-min(data))/w)
    #     -> doane: n = 1 + ceil(log(len(data), 2)) + ceil(log(1+abs(g1)/sigma)), g1 = kurtosis(data),
    #               sigma = sqrt(6*(len(data)-2)/((len(data)+1)*(len(data)+3)))
    #     -> kl: (Kullback-Leibler) glej "KL optimalno število stolpcev.ipynb"
    # n_z: samo za kl metodo - od katerega stevila stolpcev naprej gledamo
    # n_k: samo za kl metodo - do katerega stevila stolpcev gledamo
    # izris: samo za kl metodo - izris grafa KL v odvisnosti od n
    # RETURN: optimalno stevilo stolpcev = n.
    #         Pri kl metodi vrne seznam vrednosti s 5% odstopanjem (najmanjsa vrednost bo prva)
    
    if choice == "sqrt":
        n = ceil(sqrt(len(data)))
    elif choice == "scott" or choice == "scoot":
        w = 3.49*np.std(data)*len(data)**(-1/3)
        n = ceil((max(data)-min(data))/w)
    elif choice == "rice":
        n = ceil(2*len(data)**(1/3))
    elif choice == "sturges":
        n = ceil(log(len(data), 2))+1
    elif choice == "fd":
        w = 2*scipy.stats.iqr(data)*len(data)**(-1/3)
        n = ceil((max(data)-min(data))/w)
    elif choice == "doane":
        g1 = kurtosis(data, fisher=False)
        sigma = sqrt(6*(len(data)-2)/((len(data)+1)*(len(data)+3)))
        n = 1 + ceil(log(len(data), 2)) + ceil(log(1+abs(g1)/sigma))
    elif choice == "kl":
        def hist_to_fun(histogram):
            # histogram: nabor oblike (heights, edges) - stolpci in meje stolpcev histograma
            # RETURN: funkcija, ki ustreza argumentu histogram
            
            def nested(x, hist):
                [heights, edges] = hist
                # kjer ima histogram vrednost 0, mu bomo zaradi racunanja v nadaljevanju priredili vrednost eps
                h_val = np.finfo(float).eps
                for i in range(len(edges)-1):
                    if (edges[i] <= x <= edges[i+1]):
                        if heights[i] != 0:
                            h_val = heights[i]
                        break
                return h_val
            fun = lambda x: nested(x, histogram)
            return fun
            
        # KL divergenca
        def KLclen1(p, a, b):
            # p: prva gostota verjetnosti
            # q: druga gostota verjetnosti
            # a, b: meji intervala [a,b], na katerem bomo integrirali
            
            # integrand
            def integrand(x):
                return p(x)*log(p(x))         
            
            return scipy.integrate.quad(integrand, a, b)[0]

        def KLclen2(p, q, a, b):
            # p: prva gostota verjetnosti
            # q: druga gostota verjetnosti
            # a, b: meji intervala [a,b], na katerem bomo integrirali
            
            # integrand
            def integrand(x):
                return p(x)*log(q(x))         
            
            return scipy.integrate.quad(integrand, a, b)[0]


        
        # gostota verjetnosti data glede na gaussian kernel density estimation
        p = gaussian_kde(data)
        
        
        # meje integriranja
        a = min(data)
        b = max(data)

        # slovar, v katerega bomo pripenjali vrednsti KL-divergence
        KL = {}
        
        # 1. clen integrala - konstanten znotraj iteracije
        clen1 = KLclen1(p,a,b)
        
        print("iskanje od {} do {}:".format(str(n_z), str(n_k)))
        for i in range(n_z, n_k+1):
            # histogram
            hist = np.histogram(data, bins=i, density=1)
            #funkcija histograma
            h = hist_to_fun(hist)
            clen2 = KLclen2(p,h,a,b)
            div = clen1 - clen2
            KL[i] = div
            print(i, sep=' ', end=',', flush=True)
        print("\n")

        ## najmanjša vrednost
        KL_list = list(KL.items())

        KL_list.sort(key=lambda x: x[1])

        n = KL_list[0][0]

        if izris:
            
            izris_x = np.array(range(n_z, n_k+1))
            
            KL_list = list(KL.items())
            KL_list.sort(key=lambda x: x[0])

            KL_izris = [kl[1] for kl in KL_list]

            # figure
            plt.figure(figsize=(15,10))
            plt.title("KL-divergenca glede na stevilo stolpcev n")
            # graf vrednosti KL-divergence
            plt.plot(izris_x, KL_izris)
            plt.xlabel("n")
            plt.ylabel("KL")

            plt.grid()

            print(n)

    else:
        return "Ta metoda/izbira ne obstaja."

    return n


def dynamicBins(data, weights=None):
    # data: set podatkov
    # weights: utezi, ki dolocijo, kje naj bo vec stolpcev
    #          po vrsti od srednje vrednosti do repov

    if weights==None:
        weights = [0.5, 0.3, 0.2]

    # pdf
    p = gaussian_kde(data)

    # srednja vrednost in maksimum pdf
    mu = np.mean(data)
    maks = p(mu)

    # obmocja utezi
    obmocja = []
    for i in range(len(weights))[::-1]:
        obmocja.append(maks*i/len(weights))

    # definiramo funkcijo f, ki bo povedala, kaksna bo utez na intervalih pdf
    def f(x):
        for i in range(len(obmocja)):
            if obmocja[i] != 0:
                if p(x) >= obmocja[i]:
                    return weights[i]
                else:
                    continue
            else:
                if p(x) > 0:
                    return weights[i]
                else:
                    return 0

    # pogledamo, kaksne so vrednosti funkcije f na intervalu [min,max]
    vals = []
    korak = np.std(data)/10

    for i in np.arange(min(data)-korak,max(data)+korak,korak):
        vals.append((i,f(i)))

    # poiscimo intervale, kjer je vrednost funkcije f konstantna
    to_compare = vals[0]
    intervals = []
    a = to_compare[0]
    b = None

    for i in range(len(vals)-1):
        if vals[i][1] != to_compare[1]:
            b = vals[i-1][0]
            intervals.append((a,b,to_compare[1]))
            to_compare = vals[i]
            a = b
            b = None
        
    intervals.append((a,vals[-1][0],to_compare[1]))

    # dolocimo meje stolpcev nasega histograma
    stolpci = []
    optimalno_staticno = optBinNum(data) # scott metoda za dolocanje stevilo stolpcev
    first = intervals[0][0] # spodnja meja histograma
    last = intervals[-1][1] # zgornja meja histograma
    d = last - first # dolzina intervala, kjer je histogram definiran

    for el in intervals:
        st_stolpcev_na_int = ceil(2*el[2]*(el[1]-el[0])*optimalno_staticno/d) #trojka iz tistih tretjin
        array_to_append = np.linspace(el[0],el[1],st_stolpcev_na_int, endpoint=False)
        stolpci.append(array_to_append)

    stolpci = list(np.concatenate(stolpci))
    stolpci.append(intervals[-1][1])

    return stolpci