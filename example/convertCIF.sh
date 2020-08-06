# convert cif dataset to hdf5 and center and crop to resolution of 35x35
mkdir data/hdf5
python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR1_CD4+.cif data/hdf5/DONOR1_CD4+_35x35.hdf5 35
python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR1_CD8+.cif data/hdf5/DONOR1_CD8+_35x35.hdf5 35
python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR1_NK.cif data/hdf5/DONOR1_NK_35x35.hdf5 35
python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR2_CD4+.cif data/hdf5/DONOR2_CD4+_35x35.hdf5 35
python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR2_CD8+.cif data/hdf5/DONOR2_CD8+_35x35.hdf5 35
python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR2_NK.cif data/hdf5/DONOR2_NK_35x35.hdf5 35

# OPTIONALLY: cif dataset to hdf5 and apply mask to remove background
# mkdir data/hdf5
# python ../cifDataset/convertCIFtoHDF5_masked.py data/cif/DONOR1_CD4+.cif data/hdf5/DONOR1_CD4+_35x35.hdf5 35
# python ../cifDataset/convertCIFtoHDF5_masked.py data/cif/DONOR1_CD8+.cif data/hdf5/DONOR1_CD8+_35x35.hdf5 35
# python ../cifDataset/convertCIFtoHDF5_masked.py data/cif/DONOR1_NK.cif data/hdf5/DONOR1_NK_35x35.hdf5 35
# python ../cifDataset/convertCIFtoHDF5_masked.py data/cif/DONOR2_CD4+.cif data/hdf5/DONOR2_CD4+_35x35.hdf5 35
# python ../cifDataset/convertCIFtoHDF5_masked.py data/cif/DONOR2_CD8+.cif data/hdf5/DONOR2_CD8+_35x35.hdf5 35
# python ../cifDataset/convertCIFtoHDF5_masked.py data/cif/DONOR2_NK.cif data/hdf5/DONOR2_NK_35x35.hdf5 35

# convert HDF5 dataset to numpy array for faster in-memory deep learning
mkdir data/npy
python ../cifDataset/convertToNumpy.py data/hdf5/DONOR1_CD4+_35x35.hdf5 data/npy/DONOR1_CD4+_35x35.npy 35
python ../cifDataset/convertToNumpy.py data/hdf5/DONOR1_CD8+_35x35.hdf5 data/npy/DONOR1_CD8+_35x35.npy 35
python ../cifDataset/convertToNumpy.py data/hdf5/DONOR1_NK_35x35.hdf5 data/npy/DONOR1_NK_35x35.npy 35
python ../cifDataset/convertToNumpy.py data/hdf5/DONOR2_CD4+_35x35.hdf5 data/npy/DONOR2_CD4+_35x35.npy 35
python ../cifDataset/convertToNumpy.py data/hdf5/DONOR2_CD8+_35x35.hdf5 data/npy/DONOR2_CD8+_35x35.npy 35
python ../cifDataset/convertToNumpy.py data/hdf5/DONOR2_NK_35x35.hdf5 data/npy/DONOR2_NK_35x35.npy 35

# OPTIONALLY: convert HDF5 dataset to numpy array with augmented rotations
# mkdir data/npy
# python ../cifDataset/convertToNumpy_rotInvariant.py data/hdf5/DONOR1_CD4+_35x35.hdf5 data/npy/DONOR1_CD4+_35x35.npy 35
# python ../cifDataset/convertToNumpy_rotInvariant.py data/hdf5/DONOR1_CD8+_35x35.hdf5 data/npy/DONOR1_CD8+_35x35.npy 35
# python ../cifDataset/convertToNumpy_rotInvariant.py data/hdf5/DONOR1_NK_35x35.hdf5 data/npy/DONOR1_NK_35x35.npy 35
# python ../cifDataset/convertToNumpy_rotInvariant.py data/hdf5/DONOR2_CD4+_35x35.hdf5 data/npy/DONOR2_CD4+_35x35.npy 35
# python ../cifDataset/convertToNumpy_rotInvariant.py data/hdf5/DONOR2_CD8+_35x35.hdf5 data/npy/DONOR2_CD8+_35x35.npy 35
# python ../cifDataset/convertToNumpy_rotInvariant.py data/hdf5/DONOR2_NK_35x35.hdf5 data/npy/DONOR2_NK_35x35.npy 35


# OPTIONALLY: convert to different resolution
# python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR1_CD4+.cif data/hdf5/DONOR1_CD4+_40x40.hdf5 40
# python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR1_CD8+.cif data/hdf5/DONOR1_CD8+_40x40.hdf5 40
# python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR1_NK.cif data/hdf5/DONOR1_NK_40x40.hdf5 40
# python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR2_CD4+.cif data/hdf5/DONOR2_CD4+_40x40.hdf5 40
# python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR2_CD8+.cif data/hdf5/DONOR2_CD8+_40x40.hdf5 40
# python ../cifDataset/convertCIFtoHDF5.py data/cif/DONOR2_NK.cif data/hdf5/DONOR2_NK_40x40.hdf5 40

# python ../cifDataset/convertToNumpy.py data/hdf5/DONOR1_CD4+_40x40.hdf5 data/npy/DONOR1_CD4+_40x40.npy 40
# python ../cifDataset/convertToNumpy.py data/hdf5/DONOR1_CD8+_40x40.hdf5 data/npy/DONOR1_CD8+_40x40.npy 40
# python ../cifDataset/convertToNumpy.py data/hdf5/DONOR1_NK_40x40.hdf5 data/npy/DONOR1_NK_40x40.npy 40
# python ../cifDataset/convertToNumpy.py data/hdf5/DONOR2_CD4+_40x40.hdf5 data/npy/DONOR2_CD4+_40x40.npy 40
# python ../cifDataset/convertToNumpy.py data/hdf5/DONOR2_CD8+_40x40.hdf5 data/npy/DONOR2_CD8+_40x40.npy 40
# python ../cifDataset/convertToNumpy.py data/hdf5/DONOR2_NK_40x40.hdf5 data/npy/DONOR2_NK_40x40.npy 40