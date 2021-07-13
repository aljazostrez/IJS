# -*- coding: utf-8 -*-

from math import log, e
import math
import numpy as np
from scipy import integrate

## Entropija glede na gostoti verjetnosti
def renyi_entropy_cont(pdf, alpha, minimum=-float('Inf'), maximum=float('Inf')):
    '''\nReturns the renyi entropy for continuous variable.\nIf alpha=1, you must define minimum and maximum.'''
    def pdf_na_alpha(z):
        return (pdf(z)**alpha)
    def pdf_logpdf(z):
        return pdf(z)*log(pdf(z), e)

    if alpha == 1:
        try:
            return -integrate.quad(pdf_logpdf, minimum, maximum)[0]
        except TypeError:
            raise TypeError('You must define minimum and maximum')
    else:
        try:
            return (1/(1-alpha))*log(integrate.quad(pdf_na_alpha, minimum, maximum)[0], e)
        except:
            return 0
    

## Divergenca glede na gostoti verjetnosti
def renyi_divergence_cont(pdf1, pdf2, alpha, minimum, maximum):
    '''\nReturns the renyi divergence for continuous variable.
pdf1 and pdf2 must be probability density functions.
Minimum and maximum must be defined.'''
    def without_0(f,x):
        if f(x) <= np.finfo(float).eps:
            return np.finfo(float).eps
        else:
            return f(x)
    def new_pdf2(x): return without_0(pdf2, x)
    def integrand(x):
        if pdf1(x) >= np.finfo(float).eps and new_pdf2(x) >= np.finfo(float).eps:   
            return ((pdf1(x))**alpha) * ((new_pdf2(x))**(1-alpha))
        elif pdf1(x) < np.finfo(float).eps:
            return 0
        elif pdf1(x) >= np.finfo(float).eps and new_pdf2(x) < np.finfo(float).eps:
            return float('Inf')
    def KL_integrand(x):
        if pdf1(x) >= np.finfo(float).eps and new_pdf2(x) >= np.finfo(float).eps:
            return (pdf1(x)) * log((pdf1(x))/(new_pdf2(x)), e)
        elif pdf1(x) < np.finfo(float).eps:
            return 0
        elif pdf1(x) >= np.finfo(float).eps and new_pdf2(x) < np.finfo(float).eps:
            return float('Inf')

    if alpha == 1:
        return integrate.quad(KL_integrand, minimum, maximum)[0]
    else:
        return (1/(alpha-1))*log(integrate.quad(integrand, minimum, maximum)[0], e)


## Entropija glede na histogram
def renyi_hist_entropy(x, y, alpha):
    ('x - meje stolcev histograma (vključno z zadnjo mejo) \n'
     'y - višine stolpcev histograma \n'
     'alpha - red entropije \n'
     'OPOMBA: len(x) = len(y)+1'
     )
    area = 0
    if alpha == 1:
        for i in range(len(y)):
            if y[i] >= np.finfo(float).eps:
                area += y[i]*log(y[i], e)*(x[i+1]-x[i])
            else:
                pass
        return -area
    else:
        for i in range(len(y)):
            if y[i] >= np.finfo(float).eps:
                area += y[i]**alpha * (x[i+1]-x[i])
            else:
                pass
        try:
            return 1/(1-alpha)*log(area, e)
        except ValueError:
            print("ValueError")
            return 0


# Urejanje stolpcev za izracun divergence
def uredi_stolpce(x1, y1, x2, y2, zeros=1):
    ('x1 - meje stolpcev prvega histograma (vključno z zadnjo mejo) \n'
     'y1 - višine stolpcev prvega histograma (po vrsti glede na meje stolpcev) \n'
     'x2 - meje stolpcev drugega histograma (vključno z zadnjo mejo) \n'
     'y2 - višine stolpcev drugega histograma (po vrsti glede na meje stolpcev) \n'
     'zeros - 0, če naj histogrami ne vsebujejo 0, 1 če histogrami lahko vsebujejo 0. \n'
     'OPOMBA: len(x1) = len(y1)+1, len(x2) = len(y2)+1 \n'
    )
    try:
        x1 = x1.tolist()
    except:
        pass
    try:
        x2 = x2.tolist()
    except:
        pass
    try:
        y1 = y1.tolist()
    except:
        pass
    try:
        y2 = y2.tolist()
    except:
        pass
    if x1[0] < x2[0]:
        x2.insert(0, x1[0])
        y2.insert(0, 0)
    elif x2[0] < x1[0]:
        x1.insert(0, x2[0])
        y1.insert(0, 0)
    if max(x1) < max(x2):
        x1.append(max(x2))
        y1.append(0)
    elif max(x2) < max(x1):
        x2.append(max(x1))
        y2.append(0)
    x1_new = []
    x2_new = []
    y1_new = []
    y2_new = []
    for i in range(len(x1)-1):
        x1_new.append(x1[i])
        y1_new.append(y1[i])
        for el2 in x2:
            if x1[i] < el2 < x1[i+1]:
                x1_new.append(el2)
                y1_new.append(y1[i])
    for j in range(len(x2)-1):
        x2_new.append(x2[j])
        y2_new.append(y2[j])
        for el1 in x1:
            if x2[j] < el1 < x2[j+1]:
                x2_new.append(el1)
                y2_new.append(y2[j])
    x1_new.append(max(x1))
    x2_new.append(max(x2))
    x1, y1, x2, y2 = x1_new, y1_new, x2_new, y2_new
    if not zeros:
        for i in range(len(y2)):
            if y2[i] == 0:
                y2[i] = np.finfo(float).eps       
    x = x1
    return x, y1, y2




# Divergenca glede na histograma
def renyi_divergence_hist(x1, y1, x2, y2, alpha):
    ('x1 - meje stolpcev prvega histograma (vključno z zadnjo mejo) \n'
     'y1 - višine stolpcev prvega histograma (po vrsti glede na meje stolpcev) \n'
     'x2 - meje stolpcev drugega histograma (vključno z zadnjo mejo) \n'
     'y2 - višine stolpcev drugega histograma (po vrsti glede na meje stolpcev) \n'
     'OPOMBA: len(x1) = len(y1)+1, len(x2) = len(y2)+1 \n'
     'alpha - red entropije \n'
     )
    # if x1 != x2:
    bins, y1_array, y2_array = uredi_stolpce(x1, y1, x2, y2, zeros=0)
    # else:
    #     bins = x1
    y = []
    if alpha == 1:
        for i in range(len(y1_array)):
            if y1_array[i] >= np.finfo(float).eps and y2_array[i] >= np.finfo(float).eps:
                y.append(y1_array[i]*log(y1_array[i]/y2_array[i], e))
            elif y1_array[i] < np.finfo(float).eps:
                y.append(0.)
            else:
                y.append(0.)
    else:
        for i in range(len(y1_array)):
            if y1_array[i] >= np.finfo(float).eps and y2_array[i] >= np.finfo(float).eps:
                y.append(y1_array[i]**alpha * y2_array[i]**(1-alpha))
            elif y1_array[i] < np.finfo(float).eps:
                y.append(0.)
            else:
                y.append(float('Inf'))
    area = 0    
    for i in range(len(y)):
        area +=  y[i] * (bins[i+1]-bins[i])
    if alpha == 1:
        return area
    else:
        return 1/(alpha-1)*log(area, e)
    
    ###################################################################################
    ## 2D entropija in divergenca

def pomozna_matrika(bins, heigths):
    ('Metoda numpy.histogramdd podatkom priredi histogram, \n'
    'vrne pa dve vrednosti: meje stolpcev bins in visine stolpcev \n'
    'heigths.'
    )
    ar = []
    for i in range(len(heigths)):
        ar1 = []
        loop = len(heigths[i])
        for j in range(loop):
            ar1.append([bins[0][i], bins[1][j], heigths[i][j]])
        ar1.append([bins[0][i], bins[1][loop], None])
        ar.append(ar1)
    ar1 = []
    for i in range(len(bins[1])):
        m = len(bins[0])-1
        ar1.append([bins[0][m], bins[1][i], None])
    ar.append(ar1)
    return ar

def uredi_stolpce_2d(bins1, h1, bins2, h2):
    ar1 = pomozna_matrika(bins1, h1)
    ar2 = pomozna_matrika(bins2, h2)
    if ar1[0][0][0] < ar2[0][0][0]:
        first = []
        for i in range(len(ar2[0])):
            if i == len(ar2[0]) - 1:
                first.append([ar1[0][0][0], ar2[0][i][1], None])
            else:
                first.append([ar1[0][0][0], ar2[0][i][1], 0])
        ar2.insert(0, first)
    elif ar1[0][0][0] > ar2[0][0][0]:
        first = []
        for i in range(len(ar1[0])):
            if i == len(ar1[0]) - 1:
                first.append([ar2[0][0][0], ar1[0][i][1], None])
            else:
                first.append([ar2[0][0][0], ar1[0][i][1], 0])
        ar1.insert(0, first)
    if ar1[0][0][1] < ar2[0][0][1]:
        for i in range(len(ar2)):
            if i == len(ar2) - 1:
                ar2[i].insert(0, [ar2[i][0][0], ar1[0][0][1], None])
            else:
                ar2[i].insert(0, [ar2[i][0][0], ar1[0][0][1], 0])
    elif ar1[0][0][1] > ar2[0][0][1]:
        for i in range(len(ar1)):
            if i == len(ar1) - 1:
                ar1[i].insert(0, [ar1[i][0][0], ar2[0][0][1], None])
            else:
                ar1[i].insert(0, [ar1[i][0][0], ar2[0][0][1], 0])
    if ar1[len(ar1)-1][0][0] > ar2[len(ar2)-1][0][0]:
        for i in range(len(ar2[len(ar2)-1])-1):
            ar2[len(ar2)-1][i][2] = 0
        last = []
        for i in range(len(ar2[len(ar2)-1])):
            last.append([ar1[len(ar1)-1][0][0], ar2[len(ar2)-1][i][1], None])
        ar2.append(last)
    elif ar1[len(ar1)-1][0][0] < ar2[len(ar2)-1][0][0]:
        for i in range(len(ar1[len(ar1)-1])-1):
            ar1[len(ar1)-1][i][2] = 0
        last = []
        for i in range(len(ar1[len(ar1)-1])):
            last.append([ar2[len(ar2)-1][0][0], ar1[len(ar1)-1][i][1], None])
        ar1.append(last)
    if ar1[0][len(ar1[0])-1][1] > ar2[0][len(ar2[0])-1][1]:
        for i in range(len(ar2)-1):
            ar2[i][len(ar2[i])-1][2] = 0
        for i in range(len(ar2)):
            ar2[i].append([ar2[i][0][0], ar1[0][len(ar1[0])-1][1], None])
    elif ar1[0][len(ar1[0])-1][1] < ar2[0][len(ar2[0])-1][1]:
        for i in range(len(ar1)-1):
            ar1[i][len(ar1[i])-1][2] = 0
        for i in range(len(ar1)):
            ar1[i].append([ar1[i][0][0], ar2[0][len(ar2[0])-1][1], None])
    new_x = []
    new_y = []
    for i in range(len(ar1)):
        new_x.append(ar1[i][0][0])
    for j in range(len(ar2)):
        new_x.append(ar2[j][0][0])
    new_x = sorted(set(new_x))
    for i in range(len(ar1[0])):
        new_y.append(ar1[0][i][1])
    for j in range(len(ar2[0])):
        new_y.append(ar2[0][j][1])
    new_y = sorted(set(new_y))
    new_ar1 = []
    new_ar2 = []
    sez = []
    sez1 = []
    for el1 in new_x:
        for el2 in new_y:
            sez.append([el1, el2, None])
            sez1.append([el1, el2, None])
        new_ar1.append(sez)
        new_ar2.append(sez1)
        sez=[]
        sez1=[]
    for i in range(len(new_ar1)-1):
        for j in range(len(new_ar1[i])-1):
            for k in range(len(ar1)-1):
                for l in range(len(ar1[k])-1):
                    if (
                        (ar1[k][l][0] <= new_ar1[i][j][0] < ar1[k+1][l][0]) and 
                        (ar1[k][l][1] <= new_ar1[i][j][1] < ar1[k][l+1][1])
                    ):
                        new_ar1[i][j][2] = ar1[k][l][2]
    for i in range(len(new_ar2)-1):
        for j in range(len(new_ar2[i])-1):
            for k in range(len(ar2)-1):
                for l in range(len(ar2[k])-1):
                    if (ar2[k][l][0] <= new_ar2[i][j][0] <= ar2[k+1][l][0]) and (ar2[k][l][1] <= new_ar2[i][j][1] <= ar2[k][l+1][1]):
                        new_ar2[i][j][2] = ar2[k][l][2]
    ## Da nikoli ne dobimo rezultata neskoncno
    for i in range(len(new_ar2)):
        for j in range(len(new_ar2[i])):
            if new_ar2[i][j][2] == 0:
                new_ar2[i][j][2] = np.finfo(float).eps
    return new_ar1, new_ar2

## ENTROPIJA 2D
def renyi_entropy_2d(bin_edges, bin_heigths, alpha):
    ar = pomozna_matrika(bin_edges, bin_heigths)
    sum = 0
    if alpha == 1:
        for i in range(len(ar)):
            for j in range(len(ar[i])):
                try:
                    if ar[i][j][2] != None:
                        s = (ar[i+1][j][0] - ar[i][j][0]) * (ar[i][j+1][1] - ar[i][j][1]) * ar[i][j][2] * log(ar[i][j][2], e)
                        sum += s
                except ValueError:
                    pass
        return -sum
    else:
        for i in range(len(ar)):
            for j in range(len(ar[i])):
                try:    
                    if ar[i][j][2] != None:
                        s = (ar[i+1][j][0] - ar[i][j][0]) * (ar[i][j+1][1] - ar[i][j][1]) * (ar[i][j][2] ** alpha)
                        sum += s
                except ValueError:
                    pass
        return 1/(1-alpha) * log(sum, e)


## DIVERGENCA 2D
def renyi_divergence_2d(bin_edges1, bin_heigths1, bin_edges2, bin_heigths2, alpha):
    ar1, ar2 = uredi_stolpce_2d(bin_edges1, bin_heigths1, bin_edges2, bin_heigths2)
    sum = 0
    if alpha == 1:
        for i in range(len(ar1)-1):
            for j in range(len(ar1[i])-1):
                try:
                    s =  (
                        (ar1[i+1][j][0] - ar1[i][j][0]) * (ar1[i][j+1][1] - ar1[i][j][1]) * ar1[i][j][2] *
                        log(ar1[i][j][2]/ar2[i][j][2], e)
                    )
                except ValueError:
                    s = 0
                sum += s
        return sum
    else:
        for i in range(len(ar1)-1):
            for j in range(len(ar1[i])-1):
                s =  (
                    (ar1[i+1][j][0] - ar1[i][j][0]) * (ar1[i][j+1][1] - ar1[i][j][1]) * ar1[i][j][2] ** alpha *
                    ar2[i][j][2] ** (1 - alpha)
                )
                sum += s
        return 1/(alpha-1) * log(sum, e)

#######################################################################
## GENERALISED RENYI DIVERGENCE ##
#######################################################################

def generalised_RD(histograms, alpha, weigths=None):
    if weigths == None:
        weigths = [1/len(histograms) for i in histograms]
    else:
        if len(weigths) != len(histograms):
            raise ValueError("Length of the weigths list must be the same"
            "as number of the histograms"
            )
        # Natancnost vsote utezi na 5 decimalk
        if round(sum(weigths[:len(histograms)]), 5) != 1:
            raise ValueError("Sum of the weigths is not equal to 1")
    hist1 = histograms[0]
    for i in range(1, len(histograms)):
        y1,x1 = hist1
        y2,x2 = histograms[i]
        a,b,c = uredi_stolpce(x1,y1,x2,y2, zeros=1)
        hist1 = (b,a)
    y1,x1 = hist1
    histograms1 = []
    for j in range(1, len(histograms)):
        y2,x2 = histograms[j]
        a,b,c = uredi_stolpce(x1,y1,x2,y2, zeros=1)
        histograms1.append((c,a))
    histograms1.append(hist1)
    all_y = [0 for i in histograms1[0][0]]
    for i in range(len(all_y)):
        for j in range(len(histograms1)):
            try:
                all_y[i] += weigths[j] * histograms1[j][0][i]
            except IndexError:
                pass
    x = hist1[1]
    sum2 = 0
    for i in range(len(histograms)):
        sum2 += weigths[i] * renyi_hist_entropy(histograms[i][1],histograms[i][0], alpha)
    return renyi_hist_entropy(x, all_y, alpha) - sum2


def JRD_pdfs(pdfs, alpha, weigths=None, minimum=-float("inf"), maximum=float("inf")):
    def vsota(x, pdf1, pdf2, weight1=1, weight2=1):
        return weight1 * pdf1(x) + weight2 * pdf2(x)
    def nicelna(x): return 0
    if weigths == None:
        weigths = [1/len(pdfs) for i in pdfs]
    else:
        if len(weigths) != len(pdfs):
            raise ValueError("Length of the weigths list must be the same"
            "as number of the pdfs"
            )
        # Natancnost vsote utezi na 5 decimalk
        if round(sum(weigths[:len(pdfs)]), 5) != 1:
            raise ValueError("Sum of the weigths is not equal to 1")
    pdf = lambda x: vsota(x, pdfs[0], pdfs[1], weight1=weigths[0], weight2=weigths[1])
    # for i in range(len(pdfs)):
    #     pdf = lambda x: vsota(x,pdf,pdfs[i],weight1=weigths[0], weight2=weigths[1])
    #     # def dodaj(x): return vsota(x, pdf, pdfs[i], weight=weigths[i])
    #     pdf = j
    sum2 = 0
    for i in range(len(pdfs)):
        sum2 += weigths[i] * renyi_entropy_cont(pdfs[i], alpha, minimum=minimum, maximum=maximum)
    return renyi_entropy_cont(pdf, alpha, minimum=minimum, maximum=maximum) - sum2
    


#######################################################################
## RENYI DIVERGENCA 3D ##
#######################################################################

## POMOZNA MATRIKA 3D
def pomozna_matrika_3d(bins, heigths):
    ar = []
    for i in range(len(bins[0])):
        ar1 = []
        for j in range(len(bins[1])):
            ar2 = []
            for k in range(len(bins[2])):
                try:
                    ar2.append([bins[0][i], bins[1][j], bins[2][k], heigths[i][j][k]])
                except IndexError:
                    ar2.append([bins[0][i], bins[1][j], bins[2][k], None])
            ar1.append(ar2)
        ar.append(ar1)
    return ar

## UREJANJE STOLPCEV 3D
def uredi_stolpce_3d(bins1, h1, bins2, h2):
    if len(bins1)!=len(bins2) :
        return("Dimension error")
    dim = len(bins1)
    bins = {}
    for i in range(dim):
        coordinate_bins = []
        for j in range(len(bins1[i])):
            coordinate_bins.append((bins1[i][j], 0, j))
        for k in range(len(bins2[i])):
            coordinate_bins.append((bins2[i][k], 1, k))
        coordinate_bins.sort()
        bins[i] = coordinate_bins

    hist1 = []
    hist2 = []
    for el1 in bins[0]:
        ar11 = []
        ar12 = []
        for el2 in bins[1]:
            ar21 = []
            ar22 = []
            for el3 in bins[2]:
                ## 0 = None xD resena dva problema na enkrat (zadnje 
                ## vrstice se pri racunanju divergence ne doseze, je
                ## samo za sirino bina)
                ar21.append([el1[0], el2[0], el3[0], 0])
                ar22.append([el1[0], el2[0], el3[0], 0])
            ar11.append(ar21)
            ar12.append(ar22)
        hist1.append(ar11)
        hist2.append(ar12)

    ar1 = pomozna_matrika_3d(bins1,h1)
    ar2 = pomozna_matrika_3d(bins2,h2)

    lenH = len(hist1)
    lenHi = len(hist1[0])
    lenHij = len(hist1[0][0])
    for i in range(lenH-1):
        for j in range(lenHi-1):
            for k in range(lenHij-1):
                for l in range(len(ar1)-1):
                    for m in range(len(ar1[l])-1):
                        for n in range(len(ar1[l][m])-1):
                            if (
                                (ar1[l][m][n][0] <= hist1[i][j][k][0] < ar1[l+1][m][n][0]) and
                                (ar1[l][m][n][1] <= hist1[i][j][k][1] < ar1[l][m+1][n][1]) and
                                (ar1[l][m][n][2] <= hist1[i][j][k][2] < ar1[l][m][n+1][2])
                            ):
                                hist1[i][j][k][3] = ar1[l][m][n][3]
    
    for i in range(lenH-1):
        for j in range(lenHi-1):
            for k in range(lenHij-1):
                for l in range(len(ar2)-1):
                    for m in range(len(ar2[l])-1):
                        for n in range(len(ar2[l][m])-1):
                            if (
                                (ar2[l][m][n][0] <= hist2[i][j][k][0] < ar2[l+1][m][n][0]) and
                                (ar2[l][m][n][1] <= hist2[i][j][k][1] < ar2[l][m+1][n][1]) and
                                (ar2[l][m][n][2] <= hist2[i][j][k][2] < ar2[l][m][n+1][2])
                            ):
                                hist2[i][j][k][3] = ar2[l][m][n][3]
    ## da ni deljenja z 0
    for i in range(lenH):
        for j in range(lenHi):
            for k in range(lenHij):
                if hist2[i][j][k][3] == 0:
                    hist2[i][j][k][3] = np.finfo(float).eps
    return hist1, hist2

## DIVERGENCA 3D
def renyi_divergence_3d(bins1, heights1, bins2, heights2, alpha):
    hist1, hist2 = uredi_stolpce_3d(bins1, heights1, bins2, heights2)
    sum = 0
    for i in range(len(hist1)-1):
        for j in range(len(hist1[i])-1):
            for k in range(len(hist1[i][j])-1):
                if alpha != 1:
                    s = (
                        (hist1[i+1][j][k][0] - hist1[i][j][k][0]) *
                        (hist1[i][j+1][k][1] - hist1[i][j][k][1]) *
                        (hist1[i][j][k+1][2] - hist1[i][j][k][2]) *
                        hist1[i][j][k][3] ** alpha *
                        hist2[i][j][k][3] ** (1 - alpha)
                    )
                    sum += s
                else:
                    try:
                        s = (
                            (hist1[i+1][j][k][0] - hist1[i][j][k][0]) *
                            (hist1[i][j+1][k][1] - hist1[i][j][k][1]) *
                            (hist1[i][j][k+1][2] - hist1[i][j][k][2]) *
                            hist1[i][j][k][3] *
                            math.log(hist1[i][j][k][3]/hist2[i][j][k][3], math.e)
                        )
                        sum += s
                    except ValueError:
                        pass
    if alpha != 1:
        return 1/(alpha - 1) * math.log(sum, math.e)
    else:
        return sum