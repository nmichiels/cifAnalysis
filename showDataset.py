
from cifDataset.cifStreamer.hdf5Dataset import HDF5Dataset
# from cifDataset.cifStreamer.tiffDataset import TiffDataset
from cifDataset.cifStreamer.cifDataset import CIFDataset
from cifDataset.cifStreamer.npDataset_nolabels import NPDataset_nolabels
from visualizations.cvVisualizations import  visualizeDataset,visualizeDataset_withmask,visualizeDatasetGamma
import numpy as np


# examples of loading datasets
cifDataset = CIFDataset("./example/data/cif/DONOR1_CD4+.cif")
hdf5Dataset = HDF5Dataset("./example/data/hdf5/DONOR1_CD4+_35x35.hdf5")
npyDataset = NPDataset_nolabels(np.load("./example/data/npy/DONOR1_CD8+_35x35.npy"))
#tiffDataset = TiffDataset("./examples/tiffimage%d.tiff") # path to images is required, path to masks is optional


# Press any key to go to the next image. Press escape to close visualization.

# visualize dataset, works with cifDataset, hdf5Dataset, npyDataset and tiffDataset
visualizeDataset(hdf5Dataset, 35, True, 0)

# visualize dataset and masks, works with cifDataset, hdf5Dataset, tiffDataset
visualizeDataset_withmask(hdf5Dataset, 35, True, 0)

# visualize gamma corrected dataset. Only works with npy dataset because it calculates min and max intensity over entire dataset
visualizeDatasetGamma(npyDataset, 35, oneBlobsOnly = True, filterChannel = 0)


