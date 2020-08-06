import numpy as np
from cifDataset.cifStreamer.dataset import Dataset



def runTSNE_v3(data, classes, img_size, show_input = True, n_sne = 7000, n_iter = 300):
    import time
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    #%matplotlib inline

    import seaborn as sns
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    RS = 123

    # Utility function to visualize the outputs of PCA and t-SNE

    def fashion_scatter(x, colors):
        # choose a color palette with seaborn.
        num_classes = len(np.unique(colors))
        palette = np.array(sns.color_palette("hls", num_classes))

        # create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # add the labels for each digit corresponding to the label
        txts = []

        for i in range(num_classes):
            # Position of each label at median of data points.

            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"),PathEffects.Normal()])
            #txts.append(txt)
        plt.show()
        return f, ax, sc, txts

    X_train = np.reshape(data.images[:,:,:,:], (data.num_examples, img_size*img_size*data.num_channels), order='F')
    y_train = np.argmax(data.labels, axis=1)


    # Subset first 10k data points to visualize
    x_subset = X_train#[0:1000]
    y_subset = y_train#[0:1000]
    print(np.unique(y_subset))
    print(x_subset.shape)

    from sklearn.manifold import TSNE
    import time
    time_start = time.time()

    fashion_tsne = TSNE(random_state=RS).fit_transform(x_subset)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    fashion_scatter(fashion_tsne, y_subset)

   


def runTSNE_v2(data, classes, img_size, show_input = True, n_sne = 7000, n_iter = 300, maxImage = -1):
    import scanpy.api as sc
    import pandas as pd

    X_train = np.reshape(data.test.images[:,:,:,:], (data.test.num_examples, img_size*img_size*data.test.num_channels), order='F')
    y_train = np.argmax(data.test.labels, axis=1)


    

    # Subset first 10k data points to visualize
    x_subset = X_train#[0:6000]
    y_subset = y_train#[0:6000]

    if maxImage > 0:
        x_subset = X_train[0:maxImage]
        y_subset = y_train[0:maxImage]

    unique, counts = np.unique(y_subset, return_counts=True)
    print("Training Data", np.asarray((unique, counts)).T)


    labels = [classes[classID] for idx, classID in enumerate(y_subset)]
    
    adata = sc.AnnData(x_subset)
    adata.obs['cell_type'] = labels
    print(x_subset)
   
    X_tsne = sc.tl.tsne(adata,use_rep='X', learning_rate=1000)

    sc.pl.tsne(adata, color='cell_type', title='Ground truth') # Color tsne plot by ground truth cell annotations
    #plt.show()
    #plt.clf()



def runTSNE( data, classes, img_size, show_input = True, n_sne = 7000, n_iter = 300):
    from sklearn.manifold import TSNE
    import pandas as pd
    import matplotlib.pyplot as plt
    import time
    from ggplot import ggplot

    X = np.reshape(data.images[:,:,:,:], (data.num_examples, img_size*img_size*data.num_channels), order='F')
    y = np.argmax(data.labels, axis=1)


    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

    df = pd.DataFrame(X,columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))

    X, y = None, None

    print('Size of the dataframe: {}'.format(df.shape))

    rndperm = np.random.permutation(df.shape[0])

    if(show_input):
        # show some examples of the first channel in the data
        # Plot the graph
        plt.gray()
        fig = plt.figure( figsize=(16,7) )
        for i in range(0,30):
            ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )
            ax.matshow(df.loc[rndperm[i],feat_cols].values[0:img_size*img_size].reshape((img_size,img_size), order='F').astype(float))
        plt.show()




    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=n_iter)
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]
    chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point(size=70,alpha=0.1) + ggtitle("tSNE dimensions colored by Digit")
    print(chart)



def visualizeAutoencoder_v2(autoe_features_path, autoe_labels_path, tsne_features_path, title="t-SNE"):
    import scanpy.api as sc
    import pandas as pd
    import pickle
    import os
    from os import makedirs
    from os.path import exists, join

    tsne_features = None
    if type(autoe_features_path) is list:
        print('Merging features ...')
        for file in autoe_features_path:
            print(file)

        labels_list = [pickle.load(open(file, 'rb')) for file in autoe_labels_path]
        prev = None
        for labels in labels_list:
            print("Labels Shape",labels.shape)
            if prev is not None:
                if (not np.array_equal(prev, labels)):
                    print("Error: labels are not exactly the same.")
                    return
            prev = labels
        labels = labels_list[0]

        features_list = [pickle.load(open(file, 'rb')) for file in autoe_features_path]
        for features in features_list:
            print("Features Shape",features.shape)

        latent_space_list = [np.reshape(features, (features.shape[0], -1)) for features in features_list]
        for latent_space in latent_space_list:
            print("Latent Shape",latent_space.shape)

        latent_space = np.concatenate(latent_space_list, axis=1)

        #with open(join("./", 'features.tsv'), 'w') as f:
        #    np.savetxt(f, latent_space, delimiter='\t')


        print("latent_space 1D shape: ", latent_space.shape)
    else:
        if os.path.exists(autoe_labels_path):
            print('Pre-extracted labels found. Loading them ...')
            labels = pickle.load(open(autoe_labels_path, 'rb'))#[1:6000]
            print("labels shape: ", labels.shape)

        if os.path.exists(autoe_features_path):
            print('Pre-extracted features found. Loading them ...')
            latent_space = pickle.load(open(autoe_features_path, 'rb'))#[1:6000,:]
            print("latent_space shape: ", latent_space.shape)
            latentSize = np.asarray(latent_space.shape)

            latent_space = np.reshape(latent_space, (latent_space.shape[0], np.prod(latentSize[1:])))
            print("latent_space 1D shape: ", latent_space.shape)

    print(labels)
    X_train = latent_space
    y_train = labels
    
    
    # Subset first 10k data points to visualize
    x_subset = X_train#[0:6000]
    y_subset = y_train#[0:6000]

    unique, counts = np.unique(y_subset, return_counts=True)
    print("Training Data", np.asarray((unique, counts)).T)
    
    adata = sc.AnnData(x_subset)

    adata.obs['cell_type'] = labels

    X_tsne = sc.tl.tsne(adata,use_rep='X', learning_rate=1000)

    zeileis_26 = [
        "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3",
        "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593",
        "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7",
        "#f3e1eb", "#f6c4e1", "#f79cd4",
        '#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600"  # these last ones were added,
    ]


    sc.pl.tsne(adata, color='cell_type', title=title, size=20.0, color_map="tab20", palette="tab20") # Color tsne plot by ground truth cell annotations
    #plt.show()
    #plt.clf()



def visualizeAutoencoder(autoe_features_path, autoe_labels_path, tsne_features_path):
    from sklearn.manifold import TSNE


    import pandas as pd
    import matplotlib.pyplot as plt
    import pickle
    import os


    tsne_features = None

    if os.path.exists(autoe_labels_path):
        labels = pickle.load(open(autoe_labels_path, 'rb'))[1:6000]

        if os.path.exists(tsne_features_path):
            print('t-sne features found. Loading ...')
            tsne_features = pickle.load(open(tsne_features_path, 'rb'))
            
        else:
            if os.path.exists(autoe_features_path):
                print('Pre-extracted features found. Loading them ...')
                latent_space = pickle.load(open(autoe_features_path, 'rb'))[1:6000,:]
                print("latent_space shape: ", latent_space.shape)
                latentSize = np.asarray(latent_space.shape)

                latent_space = np.reshape(latent_space, (latent_space.shape[0], np.prod(latentSize[1:])))
                print("latent_space 1D shape: ", latent_space.shape)

                print('t-SNE happening ...!')
                tsne_features = TSNE(verbose=1).fit_transform(latent_space)

                pickle.dump(tsne_features, open(tsne_features_path, 'wb'))
            else:
                print('Nothing found ...')

            
       
            print(labels.shape)
            print(tsne_features.shape)
            df_tsne = pd.DataFrame()
            df_tsne['x-tsne'] = tsne_features[:,0]
            df_tsne['y-tsne'] = tsne_features[:,1]
            df_tsne['label'] = labels
            # df_tsne['labelColors'] = labelColors
            chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point( size=20,alpha=1.0) + ggtitle("tSNE dimensions colored by Digit") 
            # chart += scale_color_brewer(palette = "Set1")

            print(chart)
    else:
        print('No labels')