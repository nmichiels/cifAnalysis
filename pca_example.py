from cifDataset.cifStreamer.dataset import Dataset
from dataset.dataset import loadTestDatasetIndexed, loadTestDataset
from cnn.pca import pca
import numpy as np
#**************************************************************************
print("PCA analysis of example dataset")
#**************************************************************************


def predictAndSaveFeatures(dataFileName, datasetDir, channels, modelDir, model):
    from cnn.prediction_keras import predict as predict_keras
    data = loadTestDataset(files =   [ datasetDir + dataFileName + '.npy'],
                       classes = ['Unknown'],
                       channels = channels,
                       img_size = 35,
                       rotateInvariant=False,
                       random=True)

    
    # pure prediction of data
    features, y_pred, _ = predict_keras(data, classes = ['Unknown'], channels = channels, validation=False, returnFeatures=True, inputDir=modelDir, inputModel=model, weights_file = 'weights.best.hdf5', img_size = 35, batch_size = 256, useGenerator=False)


    np.save(modelDir + model + '/' + dataFileName + '_prediction.npy', y_pred)
    np.save(modelDir + model + '/' + dataFileName + '_features.npy', features)

    del data

    return features, y_pred


channels =  [0,1,2,3,4,5]
channelNames = ['BF', 'CD4', 'CD8', 'L/D', 'BF', 'CD3']
classes = ["CD8+","CD4+","NK"]
img_size = 35

# generate model name based on selected channels and classes
modelDir = 'example/models/'
modelName = 'CD8-CD4-NK_'
for c in channels:
    modelName = modelName + 'C%d'%c
modelName = modelName + '_valid20'


withPrediction = True # set to false if prediciton is already calculated
donor = 2
if withPrediction:
    datasetDir = "./example/data/npy/"
    features_CD8, prediction_CD8 = predictAndSaveFeatures('DONOR%d_CD8+_35x35'%donor, datasetDir, channels, modelDir, modelName)
    features_CD4, prediction_CD4 = predictAndSaveFeatures('DONOR%d_CD4+_35x35'%donor, datasetDir, channels, modelDir, modelName)
    features_NK, prediction_NK = predictAndSaveFeatures('DONOR%d_NK_35x35'%donor, datasetDir, channels, modelDir, modelName)


features_CD8 = np.load(modelDir + modelName + '/' + 'DONOR%d_CD8+_35x35_features.npy'%donor)
features_CD4 = np.load(modelDir + modelName + '/' + 'DONOR%d_CD4+_35x35_features.npy'%donor)
features_NK = np.load(modelDir + modelName + '/' + 'DONOR%d_NK_35x35_features.npy'%donor)
prediction_CD8 = np.load(modelDir + modelName + '/' + 'DONOR%d_CD8+_35x35_prediction.npy'%donor)
prediction_CD4 = np.load(modelDir + modelName + '/' + 'DONOR%d_CD4+_35x35_prediction.npy'%donor)
prediction_NK = np.load(modelDir + modelName + '/' + 'DONOR%d_NK_35x35_prediction.npy'%donor)

labels_CD8 = [ "CD8" for i in range(features_CD8.shape[0])]
labels_CD4 = [ "CD4" for i in range(features_CD4.shape[0])]
labels_NK = [ "NK" for i in range(features_NK.shape[0])]
labels = ["CD8","CD4","NK"]

x = np.concatenate([features_CD8, features_CD4, features_NK], axis=0)
y = np.concatenate([labels_CD8, labels_CD4, labels_NK], axis=0)



from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
principalComponents = sklearn_pca.fit_transform(x)


import pandas as pd
principalDf = pd.DataFrame(data = principalComponents
            , columns = ['principal component 1', 'principal component 2'])
finalDf = principalDf
finalDf['target'] = y



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = labels
if targets is None:
    targets = np.unique(y)
# colors = ['r', 'g', 'b']
colorsAll = ['#0D76BF', '#00cc96', '#EF553B', '#FFA500']
colors = colorsAll
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
            , finalDf.loc[indicesToKeep, 'principal component 2']
            , c = color
            , s = 5)
plt.ylim(-1.8,1.8)
plt.xlim(-1.8,1.8)             
ax.legend(targets)
ax.grid()
plt.savefig(modelDir + modelName + '/' + 'pcaAnalysis.png', bbox_inches='tight')

plt.show()


