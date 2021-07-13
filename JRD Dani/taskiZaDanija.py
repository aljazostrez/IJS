import csv
from tabulate import tabulate

print("\n\n\nTASK 1:\nEntropije razlicnih normalih distribucij z repi "
    "iz datoteke Normalne_z_repi_1.png glede na razlicne α.\n"
    )

entropije = []
with open("entropijeNormalnihZRepi.csv", 'r', encoding='utf-8') as dat_csv:
    reader = csv.reader(dat_csv, delimiter=',')
    header = True
    for row in reader:
        if header:
            h1 = row
            header = False
        else:
            entropije.append(row)

print(tabulate(entropije,headers=h1,tablefmt='orgtbl'))

print("\n\n\nTASK 2:\nJRD razlicnih normalnih distribucij z repi iz datotek "
    "Normalne_z_repi_1.png in Normalne_z_repi_2.png glede na razlicne α.\n"
    )

JRDi = []
with open("JRDNormalnihZRepi.csv", 'r', encoding='utf-8') as dat_csv:
    reader = csv.reader(dat_csv, delimiter=',')
    header = True
    for row in reader:
        if header:
            h2 = row
            header = False
        else:
            JRDi.append(row)

print(tabulate(JRDi,headers=h2,tablefmt='orgtbl'))

print("\n\n\nTASK 3:\nGaussian kernel density estimation beta porazdelitve"
" (predstavitev v datoteki beta_hist_pdf.png)."
)

print("\n\n\nTASK 4:\nJRD razlicnih gama distribucij iz datoteke "
    "gamma_variate_distributions.png glede na razlicne α.\n"
)

JRDi_gama = []
with open("JRDGammaDistributions.csv", 'r', encoding='utf-8') as dat_csv:
    reader = csv.reader(dat_csv, delimiter=',')
    header = True
    for row in reader:
        if header:
            h3 = row
            header = False
        else:
            JRDi_gama.append(row)

print(tabulate(JRDi_gama,headers=h3,tablefmt='orgtbl'))

print("\n\n\n")