#Predicting and visualizing classifier model for example dataset
# Data contains 6 channels
    # - Channel 1: brightfield
    # - Channel 2: CD4-FITC
    # - Channel 3: CD8-PE
    # - Channel 7: SytoxBlue (positive staining for dead cells)
    # - Channel 9: brightfield
    # - Channel 11: CD3-APC (nucleus?)


from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadTestDataset, loadTestDatasetIndexed, loadCrossValidationDataset, loadCrossValidationDatasetIndexed
#from cnn.prediction import predict, saliency, visualizeActivations, loadGraph
from cnn.prediction_keras import predict as predict_keras
from cnn.prediction_keras import extractActivationFeatures, visualizeSaliency, visualizeActivations, visualizeConvolutions

#from visualizations.embeddingTensorboard import embedding


# #**************************************************************************
print("Prediction of CD4-CD8-NK based on all 6 channels...")
# #**************************************************************************
channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]
img_size = 35
data = loadTestDatasetIndexed(files =   [ "example/data/npy/DONOR1_CD8+_35x35.npy", 
                                          "example/data/npy/DONOR1_CD4+_35x35.npy",
                                          "example/data/npy/DONOR1_NK_35x35.npy"],
                       classes = classes,
                       channels = channels,
                       img_size = img_size,
                       rotateInvariant=False)

# generate model name based on selected channels and classes
modelDir = 'example/models/'
modelName = 'CD8-CD4-NK_'
for c in channels:
    modelName = modelName + 'C%d'%c
modelName = modelName + '_valid20'



# Visualize activations per class
visualizeActivations(inputDir=modelDir, inputModel=modelName, classes=classes, channels=channels, channelNames=channelNames, weights_file = 'weights.best.hdf5', img_size=img_size)

# Visualize the convolution maps of the last convolution layer. You can change layer_name in this function to change covolution layer to visualize.
#visualizeConvolutions(inputDir=modelDir, inputModel=modelName, classes=classes, channels=channels, channelNames=channelNames, weights_file = 'weights.best.hdf5', img_size=img_size)

# Visualize attention maps per class. By pressing enter you loop through the dataset
visualizeSaliency(data, classes = classes, channels = channels, inputDir=modelDir, inputModel=modelName, weights_file = 'weights.best.hdf5', img_size = 35, channelNames=channelNames, batch_size = 256)



