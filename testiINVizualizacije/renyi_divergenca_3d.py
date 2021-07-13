import numpy as np
import math
import time

####
####
####
####
#### ZAZENI TO DATOTEKO ZA TEST 3D DIVERGENCE
####
####
####
####


## malo lepša verzija od 2d - dela
def pomozna_mat(bins, heigths):
    lenH = len(heigths)
    for i in range(lenH - 1):
        if len(heigths[i]) != len(heigths[i+1]):
            raise IndexError("Napaka v zapisu histograma.")
    lenHi = len(heigths[0])
    ar = []
    for i in range(lenH):
        ar1 = []
        ## doda min sticisca vrstic in stolpcev, katerim dodelimo visino
        for j in range(lenHi):
            ar1.append([bins[0][i], bins[1][j], heigths[i][j]])
        ## doda zadnji clen, kateremu ne dolocimo visine
        ar1.append([bins[0][i], bins[1][lenHi], None])
        ar.append(ar1)
        ## zadnja vrstica, kjer ni visin
        if i == lenH-1:
            ar1 = []
            for j in range(len(bins[1])):
                ar1.append([bins[0][lenH], bins[1][j], None])
            ar.append(ar1)
    return ar


## 4d dela!
def pomozna_matrika_4d(bins, heights):
    ar = []
    for i in range(len(bins[0])):
        ar1 = []
        for j in range(len(bins[1])):
            ar2 = []
            for k in range(len(bins[2])):
                ar3 = []
                for l in range(len(bins[3])):
                    try:
                        ar3.append([bins[0][i], bins[1][j], bins[2][k], bins[3][l], heights[i][j][k][l]])
                    except IndexError:
                        ar3.append([bins[0][i], bins[1][j], bins[2][k], bins[3][l], None])
                ar2.append(ar3)
            ar1.append(ar2)
        ar.append(ar1)
    return ar




## ta verzija je enostavna, pregledna... in dela!!!
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
                            
    
    ### sledec komentiran blok bom shranil, če mi pride kdaj prav pri
    ### rekurziji za posplošene primere
    # ## konstruiramo histogram 1 z urejenimi stolpci
    # pomozna_vrednost = 0
    # for i in range(len(bins[0])):
    #     ar1 = []
    #     for j in range(len(bins[1])):
    #         ar2 = []
    #         for k in range(len(bins[2])):
    #             x_edge = bins[0][i][0]
    #             y_edge = bins[1][j][0]
    #             z_edge = bins[2][k][0]
    #             if bins[0][i][1] == 0 and bins[1][j][1] == 0 and bins[2][k][1] == 0:
    #                 x_os = bins[0][i][2]
    #                 y_os = bins[1][j][2]
    #                 z_os = bins[2][k][2]
    #                 try:
    #                     pomozna_vrednost = h1[x_os][y_os][z_os]
    #                     ar2.append([x_edge, y_edge, z_edge, pomozna_vrednost])
    #                 except IndexError:
    #                     ar2.append([x_edge, y_edge, z_edge, None])  
    #             else:
    #                 if i == len(bins[0])-1 or j == len(bins[1])-1 or k == len(bins[2])-1:
    #                     ar2.append([x_edge, y_edge, z_edge, None])
    #                 else:
    #                     ar2.append([x_edge, y_edge, z_edge, pomozna_vrednost])
    #         ar1.append(ar2)
    #     hist1.append(ar1)

    # ## konstruiramo histogram 2 z urejenimi stolpci
    # pomozna_vrednost = 0
    # for i in range(len(bins[0])):
    #     ar1 = []
    #     for j in range(len(bins[1])):
    #         ar2 = []
    #         for k in range(len(bins[2])):
    #             x_edge = bins[0][i][0]
    #             y_edge = bins[1][j][0]
    #             z_edge = bins[2][k][0]
    #             if bins[0][i][1] == 1 and bins[1][j][1] == 1 and bins[2][k][1] == 1:
    #                 x_os = bins[0][i][2]
    #                 y_os = bins[1][j][2]
    #                 z_os = bins[2][k][2]
    #                 try:
    #                     pomozna_vrednost = h2[x_os][y_os][z_os]
    #                     ar2.append([x_edge, y_edge, z_edge, pomozna_vrednost])
    #                 except IndexError:
    #                     ar2.append([x_edge, y_edge, z_edge, None])
    #             else:
    #                 if i == len(bins[0])-1 or j == len(bins[1])-1 or k == len(bins[2])-1:
    #                     ar2.append([x_edge, y_edge, z_edge, None])
    #                 else:
    #                     ar2.append([x_edge, y_edge, z_edge, pomozna_vrednost])
    #         ar1.append(ar2)
    #     hist2.append(ar1)

    return hist1, hist2


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
    


# from nepotrebno.vecdhist import ar, nphist
# from renyi import pomozna_matrika


# h1,b1 = nphist

# ar1 = pomozna_mat(b,h)
# ar2 = pomozna_matrika(b,h)

# print(np.array(ar1))

import numpy as np
import random as r

a = []
b = []
c = []
d = []

e = [] 
f = []
g = []
h = []


for i in range(1000000):
    a.append(r.gauss(0,5))
    b.append(r.gauss(10,5))
    c.append(r.gauss(20,5))
    d.append(r.gauss(30,5))
    e.append(r.gauss(0,2))
    f.append(r.gauss(10,2))
    g.append(r.gauss(20,2))
    h.append(r.gauss(30,2))

print("generiranje končano")

h1, bins1 = np.histogramdd((a,b,c), bins=[10,10,10], density=1)
h2, bins2 = np.histogramdd((e,f,g), bins=[10,10,10], density=1)



# urejeno = uredi_stolpce_3d(bins1, h1, bins2, h2)
#
# a = urejeno[1]
# for i in range(len(a)):
#     for j in range(len(a[i])):
#         for k in range(len(a[i][j])):
#             # for l in range(len(a[i][j][k])):
#             print(a[i][j][k])
#         print("")
#         # print("\n")
#     print("\n\n")

renyi = []

for i in np.arange(0.1,3,0.1):
    start = time.time()
    renyi.append(renyi_divergence_3d(bins1, h1, bins2, h2, i))
    end = time.time() - start
    print(round(i,1), end)

import matplotlib.pyplot as plt

plt.plot(np.arange(0.1,3,0.1), renyi)

plt.grid(1)
plt.show()


