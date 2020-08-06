from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadTestDataset,loadMNISTDataset,loadCrossValidationDataset, loadCrossValidationDatasetIndexed, loadTestDatasetIndexed
# from cnn.simple_AE import train
from cnn.convolutional_AE import train,predict,train_gridSearchCV
# from cnn.deep_AE import train



## #**************************************************************************
#print("Training autoencoder on MNIST dataset")
## #**************************************************************************
#data, classes, img_size = loadMNISTDataset(fraction=0.5)
#train(  data = data,
#        classes = classes,
#        max_epochs = 20,
#        img_size = img_size,
#        outputModel  = './models/AE/model',
#        batch_size = 256,
#        output_features_path = 'conv_autoe_features_MNIST.pickle',
#        output_labels_path = 'conv_autoe_labels_MNIST.pickle',
#        logs_path = './tmp/tensorflow_logs/example/'
#    )




#**************************************************************************
print("Training autoencoder on CD4-CD8-NK based on all 6 channels...")
#**************************************************************************
channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]
image_size = 40
data = loadCrossValidationDatasetIndexed(  files =   [ "example/data/npy/DONOR1_CD8+_40x40.npy", 
                                                "example/data/npy/DONOR1_CD4+_40x40.npy",
                                                "example/data/npy/DONOR1_NK_40x40.npy"],
                                    classes = classes,
                                    channels = channels,
                                    img_size = image_size,
                                    validation_size = 0.2,
                                    rotateInvariant = False   # set to true to augment with random rotations
                                )
                                

# generate model name based on selected channels and classes
modelDir = 'example/models/'
modelName = 'AE_CD8-CD4-NK_'
for c in channels:
    modelName = modelName + 'C%d'%c
modelName = modelName + '_valid20_' + str(image_size)


train(  data = data,
        classes = classes,
        max_epochs = 200,
        img_size = image_size,
        modelNr = 2, 
        outputModel  = modelDir + modelName,
        batch_size = 256,
    )


# train_gridSearchCV(  data = data,
        # classes = classes,
        # channels = channels,
        # max_epochs = 100,
        # img_size = image_size,
        # modelNr = 2, 
        # outputModel  = modelDir,
        # batch_size = 256,
    # )

