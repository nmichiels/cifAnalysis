import numpy as np
from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadTestDataset,loadCrossValidationDataset
from visualizations.tsne import runTSNE,runTSNE_v2,runTSNE_v3


# Visualize tsne plot for example dataset
donor = 1
channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]
img_size = 35
data = loadTestDataset(  files =   [ "example/data/npy/DONOR%d_CD8+_35x35.npy"%donor, 
                                                "example/data/npy/DONOR%d_CD4+_35x35.npy"%donor,
                                                "example/data/npy/DONOR%d_NK_35x35.npy"%donor],
                                    classes = classes,
                                    channels = channels,
                                    img_size = 35,
                                    rotateInvariant = False ,  # set to true to augment with random rotations
                                    random = True
                                )


# set maximage to only select first x images
runTSNE_v2( data, classes, img_size, show_input = True, n_sne = 7000, n_iter = 300, maxImage = -1)

