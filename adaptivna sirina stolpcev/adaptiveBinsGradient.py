from math import sqrt
from scipy import integrate, stats
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

def tangentni_vektor(f, df, x0, x=1, koef=1):
    """ 
    f - funkcija;
    df - odvod funkcije;
    x0 - tocka, v kateri iscemo tangentni vektor;
    RETURN:
    (x,y) - normiran tangentni vektor;
    """

    k = df(x0)
    y = k*x
    norm = sqrt(x**2 + y**2)
    x = koef*x/norm
    y = koef*y/norm
    
    return (x,y)

def naslednji_korak(cdf, pdf, x, koef=1):
    """ 
    cdf - cumulative density function;
    pdf - probability density function;
    x - tocka;
    koef - koeficient;
    RETURN:
    dx - dolzina koraka v smeri proj_x(v);
    """
    dx = tangentni_vektor(cdf,pdf,x,koef=koef)[0]
    return dx

def dolzina_krivulje_f(df, a, b):
    """ 
    df - odvod funkcije;
    RETURN:
    d - dolzina funkcije od a do b.
    """
    d = integrate.quad(lambda x: sqrt(1+df(x)**2), a, b)
    return d[0]

def adaptiveBins(cdf,pdf,x0, koef=1, N=20,a=-3,b=3):
    ## tale je primer za gauss(0,1)
    """ 
    cdf - cumulative density function;
    pdf - probability density function;
    x0 - mode porazdelitve;
    N - stevilo tock, ki jih hocemo (cca stevilo binov);
    a - prvi bin;
    b - zadnji bin;
    RETURN:
    xs - tocke/stolpci;
    """

    cdf1 = lambda x: N*cdf(x)
    pdf1 = lambda x: N*pdf(x)
    
    x = x0
    xs = [x]
    
    # samo za unimodalne
    while x>=a:
        dx = naslednji_korak(cdf1,pdf1,x, koef=koef)
        x -= dx
        xs.append(x)
    
    x = x0
    while x<=b:
        dx = naslednji_korak(cdf1,pdf1,x, koef=koef)
        x += dx
        xs.append(x)
    
    xs = sorted(xs)
    if xs[0] < a:
        xs[0] = a
    if xs[-1] > b:
        xs[-1] = b
    
    return sorted(xs)

def data_to_adaptive_bins(data, x0=None, koef=1, N=1):
    ## samo za unimodalne porazdelitve
    if x0 == None:
        x0 = np.mean(data)
    std = np.std(data)
    
    pdf = stats.gaussian_kde(data)
    cdf = ECDF(data)

    a = min(data)
    b = max(data)
    
    return adaptiveBins(cdf,pdf,koef=koef,x0=x0,N=N,a=a,b=b)
    