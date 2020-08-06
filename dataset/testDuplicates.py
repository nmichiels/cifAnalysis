import numpy as np


def isTheSame(img1, img2):
    for row in range(img1.shape[0]):
        for col in range(img1.shape[1]):
            for chan in range(img1.shape[2]):
                if (abs(img1[row,col,chan] - img2[row,col, chan]) > 0.00001):
                    return False
    return True

data = np.load('/home/nick/ImmCyte/code/data/20190411_healthy_donor_peptide_specific_stimulation/IFNy+_CD8_CEF24_35x35.npy')

for i in range(data.shape[0]):
    img = data[i,:,:,:]
    print(i)
    for j in range(data.shape[0]):
        if (i == j):
            continue
        img2 = data[j,:,:,:]
        if isTheSame(img, img2):
            print(i, "==", j)
        #else:
            #print(i, "!=", j)

