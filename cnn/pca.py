from cifDataset.cifStreamer.dataset import Dataset
import numpy as np

def pca( data, classes, channels, img_size = 35):
    from colorama import init, Fore, Back, Style
    init(convert=True)

    num_channels = data.test.num_channels

    print(Fore.GREEN + "\nStart pca analysis...")
    print("Number of files in Test set:\t\t%6d"%(len(data.test.labels)), "[", end =" ")
    unique, counts = np.unique(np.argmax(data.test.labels, axis=1), return_counts=True)
    counts = dict(zip(unique, counts))
    for i, className in enumerate(classes):
        print(className, ":" , counts[i], end =" ")
    print("]")
    print("Number of channels to evaluate on:\t%6d"%(len(channels)), channels, "\n")
    print(Style.RESET_ALL)

    num_classes = data.test.num_classes
    numChannels = len(channels)


    x_train = data.test.images[:,:,:,:]
    y_train = np.array(classes)[np.argmax(data.test.labels, axis=1)]

     # normalize all values between 0 and 1
    x_train = x_train.astype('float32') / np.amax(x_train).astype('float32')
    #x_train = np.reshape(x_train, (len(x_train), img_size, img_size, num_channels))    # adapt this if using 'channels_first' image data format
    x_train = np.reshape(x_train, (len(x_train), -1))    # adapt this if using 'channels_first' image data format
    print(x_train.shape)

    import pandas as pd

    df = pd.DataFrame(data=y_train, columns=['target'])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Fit on training set only.
    scaler.fit(x_train)

       # Apply transform to both the training set and the test set.
    x_train = scaler.transform(x_train)

    from sklearn.decomposition import PCA
    # Make an instance of the Model
    pca = PCA(n_components=2)#PCA(.95)

    pca.fit(x_train)
    print("Number of coeffs PCA: ", pca.n_components_)



    x_train_pca = pca.transform(x_train)

    print(x_train_pca[:,1:].shape)
    

    
    
    principalDf = pd.DataFrame(data = x_train_pca[:,0:2]
             , columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
    print(finalDf)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = classes
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        print(target)
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 5)
    ax.legend(targets)
    ax.grid()
    plt.show()
