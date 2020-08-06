#Training classifier for distinguishing between CD8 and CD4 T-cells
# Data contains 6 channels
    # - Channel 1: brightfield
    # - Channel 2: CD4-FITC
    # - Channel 3: CD8-PE
    # - Channel 7: SytoxBlue (positive staining for dead cells)
    # - Channel 9: brightfield
    # - Channel 11: CD3-APC (nucleus?)
from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadCrossValidationDataset, loadMNISTDataset, loadCrossValidationDatasetIndexed
from cnn.train_siamese_keras import train as train_siamese


                    
channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]
img_size = 35
# load the cross validation data
data = loadCrossValidationDataset(  files =   [ "example/data/npy/DONOR1_CD8+_35x35.npy", 
                                                "example/data/npy/DONOR1_CD4+_35x35.npy",
                                                "example/data/npy/DONOR1_NK_35x35.npy"],
                                    classes = classes,
                                    channels = channels,
                                    img_size = img_size,
                                    validation_size = 0.2,
                                    rotateInvariant = False,   # set to true to augment with random rotations
                                    random = True
                                )
                                
                                
# generate model name based on selected channels and classes
modelDir = 'example/models/'
modelName = 'siamese_CD8-CD4-NK_'
for c in channels:
    modelName = modelName + 'C%d'%c
modelName = modelName + '_valid20'


                                
train_siamese(  data = data, 
        classes = classes,
        channels = channels,
        max_iterations = 300000,
        max_epochs = 100,
        img_size = img_size,
        outputDir  = modelDir,
        outputModel = modelName,
        batch_size = 256,
        permutate = True,
        validation_size = 0.2
    )
