import pickle
import numpy as np
from renyi import uredi_stolpce, renyi_divergence_hist, generalised_RD
import matplotlib.pyplot as plt

loaded_results = pickle.load(open('Aljaz/cea_all_parameter_pdfs_3poles_20bins_Cell3_only.pkl', "rb" ))
# loaded_results = pickle.load(open('something.pkl', 'rb'))

print(loaded_results.keys())
print("\n")
# print(loaded_results['R1_mu'].shape)
dol,_= loaded_results['R1_mu'].shape
# print("\n")

# Cell 1 is from 0 - dol/6
# Cell 2 is from dol/6 - 2*dol/6
# Cell 3 is from 2*dol/6 - 3*dol/6
# Cell 4 is from 3*dol/6 - 4*dol/6
# Cell 5 is from 4*dol/6 - 5*dol/6
# Cell 6 is from 5*dol/6 - 6*dol/6


# print(dol)

# for i in range(10):
#     print("{}/{}".format(str(i), str(len(loaded_results["R1_pdf"]))))
#     plt.figure()
#     plt.plot(loaded_results['R1_bin'][i], loaded_results['R1_pdf'][i])
# plt.show()


cell = {}
for i in range(6):
    cell[i] = {}
    for key in loaded_results.keys():
        # if "pdf" in key:
        #     pdf = loaded_results[key][int(i*dol/6):int((i+1)*dol/6)]
        #     cell[i][key]= pdf
        # elif "bin" in key:
        #     bins = loaded_results[key][int(i*dol/6):int((i+1)*dol/6)]
        #     cell[i][key] = bins
        value = loaded_results[key][int(i*dol/6):int((i+1)*dol/6)]
        cell[i][key]= value



def marginalni_pdf(celica, meritev):
    plt.figure("Marginalni histogrami, celica {}, meritev {}/511".format(str(celica), str(meritev)), figsize=(20,10))

    plt.subplot(4,3,1)
    plt.plot(cell[celica]["Rs_bin"][meritev],cell[celica]["Rs_pdf"][meritev])
    plt.title("Rs")
    plt.subplot(4,3,2)
    plt.plot(cell[celica]["R1_bin"][meritev],cell[celica]["R1_pdf"][meritev])
    plt.title("R1")
    plt.subplot(4,3,3)
    plt.plot(cell[celica]["R2_bin"][meritev],cell[celica]["R2_pdf"][meritev])
    plt.title("R2")
    plt.subplot(4,3,4)
    plt.plot(cell[celica]["R3_bin"][meritev],cell[celica]["R3_pdf"][meritev])
    plt.title("R3")
    plt.subplot(4,3,5)
    plt.plot(cell[celica]["tau1_bin"][meritev],cell[celica]["tau1_pdf"][meritev])
    plt.title("tau1")
    plt.subplot(4,3,6)
    plt.plot(cell[celica]["tau2_bin"][meritev],cell[celica]["tau2_pdf"][meritev])
    plt.title("tau2")
    plt.subplot(4,3,7)
    plt.plot(cell[celica]["tau3_bin"][meritev],cell[celica]["tau3_pdf"][meritev])
    plt.title("tau3")
    plt.subplot(4,3,8)
    plt.plot(cell[celica]["alfa1_bin"][meritev],cell[celica]["alfa1_pdf"][meritev])
    plt.title("alfa1")
    plt.subplot(4,3,9)
    plt.plot(cell[celica]["alfa2_bin"][meritev],cell[celica]["alfa2_pdf"][meritev])
    plt.title("alfa2")
    plt.subplot(4,3,10)
    plt.plot(cell[celica]["alfa3_bin"][meritev],cell[celica]["alfa3_pdf"][meritev])
    plt.title("alfa3")
    plt.subplot(4,3,11)
    plt.plot(cell[celica]["sigma_bin"][meritev],cell[celica]["sigma_pdf"][meritev])
    plt.title("sigma")
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.54, wspace=0.27)

    plt.show()


def marginalni_histogrami(celica, meritev):
    plt.figure("Marginalni histogrami, celica {}, meritev {}/511".format(str(celica), str(meritev)), figsize=(20,10))

    # plt.subplot(4,3,1)
    # w = cell[celica]["Rs_bin"][meritev][1]-cell[celica]["Rs_bin"][meritev][0]
    # plt.bar(x=cell[celica]["Rs_bin"][meritev],height=cell[celica]["Rs_pdf"][meritev], width=w)
    # plt.title("Rs")
    plt.subplot(3,3,2-1)
    w = cell[celica]["R1_bin"][meritev][1]-cell[celica]["R1_bin"][meritev][0]
    plt.bar(x=cell[celica]["R1_bin"][meritev], height=cell[celica]["R1_pdf"][meritev], width=w)
    plt.title("R1")
    plt.subplot(3,3,3-1)
    w = cell[celica]["R2_bin"][meritev][1]-cell[celica]["R2_bin"][meritev][0]
    plt.bar(x=cell[celica]["R2_bin"][meritev],height=cell[celica]["R2_pdf"][meritev], width=w)
    plt.title("R2")
    plt.subplot(3,3,4-1)
    w = cell[celica]["R3_bin"][meritev][1]-cell[celica]["R3_bin"][meritev][0]
    plt.bar(x=cell[celica]["R3_bin"][meritev],height=cell[celica]["R3_pdf"][meritev], width=w)
    plt.title("R3")
    plt.subplot(3,3,5-1)
    w = cell[celica]["tau1_bin"][meritev][1]-cell[celica]["tau1_bin"][meritev][0]
    plt.bar(x=cell[celica]["tau1_bin"][meritev],height=cell[celica]["tau1_pdf"][meritev], width=w)
    plt.title("tau1")
    plt.subplot(3,3,6-1)
    w = cell[celica]["tau2_bin"][meritev][1]-cell[celica]["tau2_bin"][meritev][0]
    plt.bar(x=cell[celica]["tau2_bin"][meritev],height=cell[celica]["tau2_pdf"][meritev], width=w)
    plt.title("tau2")
    plt.subplot(3,3,7-1)
    w = cell[celica]["tau3_bin"][meritev][1]-cell[celica]["tau3_bin"][meritev][0]
    plt.bar(x=cell[celica]["tau3_bin"][meritev],height=cell[celica]["tau3_pdf"][meritev], width=w)
    plt.title("tau3")
    plt.subplot(3,3,8-1)
    w = cell[celica]["alfa1_bin"][meritev][1]-cell[celica]["alfa1_bin"][meritev][0]
    plt.bar(x=cell[celica]["alfa1_bin"][meritev],height=cell[celica]["alfa1_pdf"][meritev], width=w)
    plt.title("alfa1")
    plt.subplot(3,3,9-1)
    w = cell[celica]["alfa2_bin"][meritev][1]-cell[celica]["alfa2_bin"][meritev][0]
    plt.bar(x=cell[celica]["alfa2_bin"][meritev],height=cell[celica]["alfa2_pdf"][meritev], width=w)
    plt.title("alfa2")
    plt.subplot(3,3,9)
    w = cell[celica]["alfa3_bin"][meritev][1]-cell[celica]["alfa3_bin"][meritev][0]
    plt.bar(x=cell[celica]["alfa3_bin"][meritev],height=cell[celica]["alfa3_pdf"][meritev], width=w)
    plt.title("alfa3")
    # plt.subplot(4,3,11)
    # w = cell[celica]["sigma_bin"][meritev][1]-cell[celica]["sigma_bin"][meritev][0]
    # plt.bar(x=cell[celica]["sigma_bin"][meritev],height=cell[celica]["sigma_pdf"][meritev], width=w)
    # plt.title("sigma")
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.54, wspace=0.27)

    plt.show()
