
from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadCrossValidationDataset, loadMNISTDataset, loadCrossValidationDatasetIndexed
from clustering.ae_clustering import cluster

channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]
image_size = 35
# load the cross validation data
data = loadCrossValidationDatasetIndexed(  files =   [ "example/data/npy/DONOR1_CD8+_35x35.npy", 
                                                "example/data/npy/DONOR1_CD4+_35x35.npy",
                                                "example/data/npy/DONOR1_NK_35x35.npy"],
                                    classes = classes,
                                    channels = channels,
                                    img_size = image_size,
                                    validation_size = 0.2,
                                    rotateInvariant = False   # set to true to augment with random rotations
                                )


modelDir = 'example/models/'
modelName = 'deepclustering_CD8-CD4-NK_'
for c in channels:
    modelName = modelName + 'C%d'%c
modelName = modelName + '_valid20'


cluster(  data = data, 
        classes = classes,
        channels = channels,
        max_iterations = 300000,
        max_epochs = 300,
        img_size = image_size,
        outputDir  = modelDir,
        outputModel = modelName,
        batch_size = 256,
        permutate = True,
        validation_size = 0.2
    )
