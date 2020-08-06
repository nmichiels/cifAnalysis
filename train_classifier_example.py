#Example training classifier for distinguishing between CD8 and CD4 and NK T-cells
# Data contains 6 channels
    # - Channel 1: brightfield
    # - Channel 2: CD4-FITC
    # - Channel 3: CD8-PE
    # - Channel 7: SytoxBlue (positive staining for dead cells)
    # - Channel 9: brightfield
    # - Channel 11: CD3-APC (nucleus?)
from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadCrossValidationDataset, loadCrossValidationDatasetIndexed
from cnn.training import train
from cnn.training_keras import train as train_keras
from cnn.training_keras import train_plotLearningRates


#**************************************************************************
print("Training CD4-CD8-NK based on all 6 channels...")
#**************************************************************************
channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]

# load the cross validation data
data = loadCrossValidationDatasetIndexed(  files =   [ "example/data/npy/DONOR1_CD8+_35x35.npy", 
                                                "example/data/npy/DONOR1_CD4+_35x35.npy",
                                                "example/data/npy/DONOR1_NK_35x35.npy"],
                                    classes = classes,
                                    channels = channels,
                                    img_size = 35,
                                    validation_size = 0.2,
                                    rotateInvariant = False   # set to true to augment with random rotations
                                )



## When the datasets are to large to keep in memory, the user should use a hdf5 dataset instead of a numpy as shown below
#data = loadCrossValidationDatasetIndexed(  files =   [ "example/data/hdf5/DONOR1_CD8+_35x35.hdf5", 
#                                                "example/data/hdf5/DONOR1_CD4+_35x35.hdf5",
#                                                "example/data/hdf5/DONOR1_NK_35x35.hdf5"],
#                                    classes = classes,
#                                    channels = channels,
#                                    img_size = 35,
#                                    rotateInvariant = False  # set to true to augment with random rotations
#                                )



# generate model name based on selected channels and classes
modelDir = 'example/models/'
modelName = 'CD8-CD4-NK_'
for c in channels:
    modelName = modelName + 'C%d'%c
modelName = modelName + '_valid20'

# standard training of the model
train_keras(  data = data, 
      classes = classes,
      channels = channels,
      max_iterations = 300000,
      max_epochs = 200,
      img_size = 35,              # image size
      outputDir  = modelDir,      # ouput dir for model
      outputModel = modelName,    # output name for model
      batch_size = 256,
      permutate = True,           # randomly shuffle all data
      augmentation = False,       # boolean to enable data augmentation (random translations, rotations, zooms, noise, contrast, etc...) Values can be changed in train_keras
      addGaussianNoise = False,   # add gaussian noise layer to the model
      masking = False,
      residual = False,           # use residual network, does not work as expected
      validation_size = 0.2,      # percentage of data to use as validation
      display_epoch = 1,
      dropout = 0.25,             # define dropout percentage
      initialWeights = None       # pass initial weights
  )


# Alternative: Train multiple times with a different range of learning rates. The loss, accuracy, validation loss and validation accuracy are written as csv into the output model directory.
# train_plotLearningRates(  data = data, 
        # classes = classes,
        # channels = channels,
        # max_iterations = 300000,
        # max_epochs = 50,
        # img_size = 35,
        # outputDir  = modelDir,
        # outputModel = modelName + '_learningRates',
        # batch_size = 256,
        # permutate = True,
        # validation_size = 0.2
    # )



# Alternative: grid search for hyper parameter tuning
from cnn.training_keras import train_gridSearchCV
train_gridSearchCV(  data = data, 
    classes = classes,
    channels = channels,
    max_iterations = 300000,
    max_epochs = 100,
    img_size = 35,
    outputDir  = modelDir,
    outputModel = modelName + "_gridSearch",
    batch_size = 256,
    permutate = True,
    validation_size = 0.2
)
