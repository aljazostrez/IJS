import numpy as np
from math import sqrt, log, ceil
from scipy.stats import kurtosis, iqr, gaussian_kde
import scipy
from manual_integration import integrate_num
import matplotlib.pyplot as plt
from adaptiveBinsGradient import data_to_adaptive_bins

def KL_adaptivni(data, n_z=5, n_k=100, izris=False):
    
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
        xs = data_to_adaptive_bins(data, N=i)
        hist = np.histogram(data, bins=xs, density=1)
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

    xs = data_to_adaptive_bins(data, N=n)
    
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

        # sprejemljiva napaka - 5% na [min,max] intervalu
        a = 1.05*min(KL_izris)
        y = [a for x in izris_x]
        print(a, min(KL_izris))

        plt.plot(izris_x, y)
        plt.grid()

        sprejemljivi = []
        for el in KL_izris:
            if el <= a: sprejemljivi.append(KL_izris.index(el)+n_z)

        print("Indeksi pod mejo napake:\n")
        print(sprejemljivi)
        print("")
        
    return xs