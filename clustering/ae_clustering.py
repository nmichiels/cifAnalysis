from cifDataset.cifStreamer.dataset import Dataset
import numpy as np
import keras
import time
from .metrics import acc as metrics_acc
from .metrics import nmi as metrics_nmi
from .metrics import ari as metrics_ari

from keras.utils import to_categorical
# from .keras_utils import My_Generator, My_GeneratorIncremental

import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


def cluster(  data,
            classes,
            channels,
            max_iterations,
            max_epochs,
            img_size,
            outputDir,
            outputModel,
            batch_size = 256,
            permutate = True,
            masking = False,
            validation_size = 0.2,
            display_epoch = 1,
            dropout = 0.25,
            logs_path = './tmp/tensorflow_logs/example/'
        ):
    
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Flatten,Dropout,MaxPooling2D
    from keras.callbacks import ModelCheckpoint
    from colorama import Fore, Back, Style
    import os
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)

    print(Fore.GREEN + "\nStart training", max_epochs, "epochs...")
    print("Number of files in Training-set:\t\t%6d"%(len(data.train.labels)), "[", end =" ")
    unique, counts = np.unique(np.argmax(data.train.labels, axis=1), return_counts=True)
    counts = dict(zip(unique, counts))
    for i, className in enumerate(classes):
        print(className, ":" , counts[i], end =" ")
    print("]")

    print("Number of files in Validation-set:\t\t%6d"%(len(data.test.labels)), "[", end =" ")
    unique, counts = np.unique(np.argmax(data.test.labels, axis=1), return_counts=True)
    counts = dict(zip(unique, counts))
    for i, className in enumerate(classes):
        print(className, ":" , counts[i], end =" ")
    print("]")
    print("Number of channels to train on:\t\t\t%6d"%(len(channels)), channels, "\n")
    print(Style.RESET_ALL)

    num_classes = data.train.num_classes
    num_channels = len(channels)#data.train.num_channels


    [x_train, y_train] =  data.train.nextBatch(data.train.num_examples, img_size)
    [x_test, y_test] =  data.test.nextBatch(data.test.num_examples, img_size)


    from sklearn.cluster import KMeans
    # np.random.seed(10)

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    y_cls = np.argmax(y, axis=1)

    x = x.reshape((x.shape[0], -1))
    # x = np.divide(x, 255.)
    n_clusters = num_classes
    

    # from keras.datasets import mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x = np.concatenate((x_train, x_test))
    # y = np.concatenate((y_train, y_test))
    # y_cls = y
    # x = x.reshape((x.shape[0], -1))
    # x = np.divide(x, 255.)
    # n_clusters = len(np.unique(y))

    dims = [x.shape[-1], 500, 500, 2000, 10]
    print("dims:", x.shape)
    init = VarianceScaling(scale=1. / 3., mode='fan_in',
                            distribution='uniform')
    pretrain_optimizer = SGD(lr=1, momentum=0.9)
    pretrain_epochs = 30
    batch_size = 256
    save_dir = outputDir + outputModel

    if not os.path.exists(save_dir):
            os.makedirs(save_dir)



    
    autoencoder, encoder = build_autoencoder(dims, init=init)


    pretrain = True
    if pretrain:
        from keras.utils import plot_model
        #plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
        #from IPython.display import Image
        #Image(filename='autoencoder.png')
        print(x.shape)
        print(y.shape)
        autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
        autoencoder.save_weights(save_dir + '/ae_weights.h5')

    autoencoder.load_weights(save_dir + '/ae_weights.h5')
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # computing an auxiliary target distribution
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    loss = 0
    train_deepclustering = True
    if train_deepclustering:
        
        index = 0
        maxiter = 8000
        update_interval = 140
        index_array = np.arange(x.shape[0])
        tol = 0.001 # tolerance threshold to stop training


        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = model.predict(x, verbose=0)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics_acc(y_cls, y_pred), 5)
                    nmi = np.round(metrics_nmi(y_cls, y_pred), 5)
                    ari = np.round(metrics_ari(y_cls, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

        model.save_weights(save_dir + '/DEC_model_final.h5')

    model.load_weights(save_dir + '/DEC_model_final.h5')
    # Eval.
    q = model.predict(x, verbose=0)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)
    if y_cls is not None:
        acc = np.round(metrics_acc(y_cls, y_pred), 5)
        nmi = np.round(metrics_nmi(y_cls, y_pred), 5)
        ari = np.round(metrics_ari(y_cls, y_pred), 5)
        loss = np.round(loss, 5)
        print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

    import seaborn as sns
    import sklearn.metrics
    import matplotlib.pyplot as plt
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_cls, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()
    # x = x.reshape((x.shape[0], -1))
    # x = np.divide(x, 255.)
    # # 10 clusters
    # n_clusters = num_classes
    # # Runs in parallel 4 CPUs
    # kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    # # Train K-Means.
    # y_pred_kmeans = kmeans.fit_predict(x)

    # print(y_pred_kmeans.shape)
    # print(y_cls.shape)
    # # Evaluate the K-Means clustering accuracy.

    
    # print(acc(y_cls, y_pred_kmeans))
    # y_pred_cls = y_pred_kmeans
    # print(y_pred_cls)

    # import seaborn as sns
    # import sklearn.metrics
    # import matplotlib.pyplot as plt
    # sns.set(font_scale=3)
    # confusion_matrix = sklearn.metrics.confusion_matrix(y_cls, y_pred_cls)

    # plt.figure(figsize=(16, 14))
    # sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
    # plt.title("Confusion matrix", fontsize=30)
    # plt.ylabel('True label', fontsize=25)
    # plt.xlabel('Clustering label', fontsize=25)
    # plt.show()



