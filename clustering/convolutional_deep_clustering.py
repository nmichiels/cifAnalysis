# Based on: https://github.com/XifengGuo/DEC-keras

from dataset.dataset import DataSet
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras import callbacks
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras.metrics as metrics
from keras import losses

from time import time
from keras.engine.topology import Layer, InputSpec
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import normalized_mutual_info_score
from keras.callbacks import TensorBoard

import signal

def Encoder(input_img):
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Reshape((128,))(x) # flatten 4x4x8 to 128
    return Model(input_img, encoded, name="Encoder")



def Decoder_32(input_img):  
    x = Reshape((4,4,8))(input_img) # reverse flatten 128 to 4x4x8
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded, name="Decoder")


def Decoder_lessThan32(input_img):  
    x = Reshape((4,4,8))(input_img) # reverse flatten 128 to 4x4x8
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded, name="Decoder")





def buildAutoencoder(img_size, num_channels):
    # define input to the encoder and decoder model:
    input_img = Input(shape=(img_size, img_size, num_channels))
    encoded_img = Input(shape=(128,)) #Input(shape=(4, 4, 8))

    # make the encoder, decoder and autoencoder models:
    encoder = Encoder(input_img)

    if (img_size > 32):
        print("Only datasets of 32 resolution or less are supporten.")
        return
    elif (img_size == 32):
        decoder = Decoder_32(encoded_img)
    else:
        decoder = Decoder_lessThan32(encoded_img)

    autoencoder = Model(input_img, decoder(encoder(input_img)), name="Autoencoder")


    autoencoder.compile(optimizer='adadelta', loss=losses.binary_crossentropy, metrics=['accuracy'])
    return autoencoder, encoder, decoder

def pretrain(data,
            classes,
            max_epochs,
            img_size = 32,
            # outputModel  = './models/model/model',
            batch_size = 256,
            continue_training = False,
            save_dir = './results',
            output_features_path = 'convolutional_deep_clustering_pretrain_features.pickle',
            output_labels_path = 'convolutional_deep_clustering_pretrain_labels.pickle'
            # logs_path = '/tmp/tensorflow_logs/example/'
):
    num_channels = data.train.num_channels

    
    [autoencoder, encoder, decoder] = buildAutoencoder(img_size, num_channels)
    autoencoder.summary()

    

    x_train = data.train.images[:,:,:,:]
    x_test = data.test.images[:,:,:,:]

    y_train = np.array(classes)[np.argmax(data.train.labels, axis=1)]
    y_test = np.array(classes)[np.argmax(data.test.labels, axis=1)]

     # normalize all values between 0 and 1
    x_train = x_train.astype('float32') / np.amax(x_train).astype('float32')
    x_test = x_test.astype('float32') / np.amax(x_test).astype('float32')
    x_train = np.reshape(x_train, (len(x_train), img_size, img_size, num_channels))    # adapt this if using 'channels_first' image data format
    x_test = np.reshape(x_test, (len(x_test), img_size, img_size, num_channels))       # adapt this if using 'channels_first' image data format
    print(x_train.shape)
    print(x_test.shape)


    # x = np.concatenate((x_train, x_test))
    # y = np.concatenate((y_train, y_test))
    # x = x.reshape((x.shape[0], -1))

    print('Data samples', x_train.shape)

    
    if (continue_training):
        autoencoder.load_weights(save_dir + '/ae_weights.h5')

    
    # encoded_imgs = encoder.predict(x_train)
    # decoded_imgs = decoder.predict(encoded_imgs)

    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(3, 10, i+1)
    #     plt.imshow(x_test[i].reshape(img_size, img_size,num_channels)[:,:,0])
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # display reconstruction
    #     ax = plt.subplot(3, 10, n + i+1)
    #     plt.imshow(decoded_imgs[i].reshape(img_size, img_size,num_channels)[:,:,0])
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # # display masks
    #     # ax = plt.subplot(3, 10, 2*n + i+1)
    #     # plt.imshow(x_test_mask[i].reshape(img_size, img_size))
    #     # plt.gray()
    #     # ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    # plt.show()


    autoencoder.fit(x_train, x_train,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')], verbose=2)
    
    autoencoder.save_weights(save_dir + '/ae_weights.h5')
    
    encoded_imgs = encoder.predict(x_train)
    decoded_imgs = decoder.predict(encoded_imgs)

    pickle.dump(encoded_imgs, open(output_features_path, 'wb'))
    pickle.dump(y_train, open(output_labels_path, 'wb'))

    # n = 10
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(3, 10, i+1)
    #     plt.imshow(x_train[i].reshape(img_size, img_size,num_channels)[:,:,0])
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # display reconstruction
    #     ax = plt.subplot(3, 10, n + i+1)
    #     plt.imshow(decoded_imgs[i].reshape(img_size, img_size,num_channels)[:,:,0])
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # # display masks
    #     # ax = plt.subplot(3, 10, 2*n + i+1)
    #     # plt.imshow(x_test_mask[i].reshape(img_size, img_size))
    #     # plt.gray()
    #     # ax.get_xaxis().set_visible(False)
    #     # ax.get_yaxis().set_visible(False)
    # plt.show()



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
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        print("super")
        super(ClusteringLayer, self).__init__(**kwargs)
        print("end super")
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        print("end init")

    def build(self, input_shape):
        print("build")
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        print("call")
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
    
def cluster_acc(y_true, y_pred):
    '''
    Uses the hungarian algorithm to find the best permutation mapping and then calculates the accuracy wrt
    Implementation inpired from https://github.com/piiswrong/dec, since scikit does not implement this metric
    this mapping and true labels
    :param y_true: True cluster labels
    :param y_pred: Predicted cluster labels
    :return: accuracy score for the clustering
    '''
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int32)
    for i in range(y_pred.size):
        idx1 = int(y_pred[i])
        idx2 = int(y_true[i])
        w[idx1, idx2] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def getClusterMetricString(method_name, labels_true, labels_pred):
    '''
    Creates a formatted string containing the method name and acc, nmi metrics - can be used for printing
    :param method_name: Name of the clustering method (just for printing)
    :param labels_true: True label for each sample
    :param labels_pred: Predicted label for each sample
    :return: Formatted string containing metrics and method name
    '''
    acc = cluster_acc(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return '%-40s ACC: %8.3f NMI: %8.3f' % (method_name, acc, nmi)


def evaluateKMeans(data, labels, nclusters, method_name):
    '''
    Clusters data with kmeans algorithm and then returns the string containing method name and metrics, and also the evaluated cluster centers
    :param data: Points that need to be clustered as a numpy array
    :param labels: True labels for the given points
    :param nclusters: Total number of clusters
    :param method_name: Name of the method from which the clustering space originates (only used for printing)
    :return: Formatted string containing metrics and method name, cluster centers
    '''
    kmeans = KMeans(n_clusters=nclusters, n_init=20)
    kmeans.fit(data)
    return getClusterMetricString(method_name, labels, kmeans.labels_), kmeans.cluster_centers_

def getKMeansLoss(latent_space_expression, soft_assignments, t_cluster_centers, num_clusters, latent_space_dim, num_samples, soft_loss=False):
            num_samples = K.shape(latent_space_expression)[0]
            # print(num_samples)
            # Kmeans loss = weighted sum of latent space representation of inputs from the cluster centers
            z = K.reshape(latent_space_expression, (-1, 1, latent_space_dim))
            z = K.tile(z, (1, num_clusters, 1))
            # u = Reshape((-1, 1, latent_space_dim))(t_cluster_centers)
            # print(u)
            u = K.expand_dims(   t_cluster_centers,   axis=0)
            u = K.reshape(u, (-1, num_clusters, latent_space_dim))
            u = K.tile(u, (num_samples, 1, 1))
        

            distances = K.square(K.sum(K.pow((z - u), 2), axis=2))
            
            # print(soft_assignments)
            if soft_loss:
                weighted_distances = distances * soft_assignments
                loss = K.mean(K.sum(weighted_distances, axis=1))
            else:
                # assignments = (soft_assignments == soft_assignments.max(axis=1)[:,None]).astype(float)

                # weighted_distances = distances * assignments
                # loss = K.mean(K.max(weighted_distances, axis=1))
                # # loss = K.mean(K.min(distances, axis=1))
                loss = K.mean(K.min(distances, axis=1))
            return loss

def unitTests():
    print("1D Unit Test KMeans Loss")
    # kmeansTest = KMeans(n_clusters=2, n_init=20)
    # kmeansTest.fit()
    
    clusterC = np.array([[0, 0],[5, 4]])
    x = Input(shape=(2,))
    softassignments = np.array([[0],[1],[1]])

    cc = K.variable(value=clusterC)
    loss_func = K.Function([x], [getKMeansLoss(x, softassignments, cc, 2, 2, 3, False)]) 
    loss_func_soft = K.Function([x], [getKMeansLoss(x, softassignments, cc, 2, 2, 3, True)]) 

    inputData = [[[0, 0],[5, 4],[5, 4]]]


    assert loss_func(inputData) == [[0]]
    # assert ((loss_func_soft(inputData) < [[1120.666667]]) and (loss_func_soft(inputData)  > [[1120.666665]]))

    inputData = [[[0, 1],[6, 4],[5, 3]]]

    assert loss_func(inputData) == [[1]]
    # assert loss_func_soft(inputData) == [[1]]


    

def getKLDivLossExpression(Q_expression, P_expression):
        # Loss = KL Divergence between the two distributions
        log_arg = P_expression / Q_expression
        log_exp = K.log(log_arg)
        sum_arg = P_expression * log_exp
        loss = K.sum(K.sum(sum_arg, axis=1), axis=0)
        return loss

shouldStopNow = False

def signal_handler(signal, frame):
    global shouldStopNow
    command = input('\nWhat is your command?')
    if str(command).lower()=="stop":
        shouldStopNow = True
    else:
        exec(command)




class ClusterAccuracyCallback(callbacks.Callback):
    def __init__(self, training_data):
        self.training_data = training_data
        self.qij = []
        self.losses = []
        self.clusterAcc = []
        return

    def on_train_begin(self, logs={}):
        # self.aucs = []
        return
        

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(np.round(logs.get('loss'), 5))

        labels_pred = self.qij.argmax(axis=1)
        labels_true = self.training_data[1]
        acc = np.round(cluster_acc(labels_true, labels_pred), 5)
        self.clusterAcc.append(acc)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)
        print('\nCluster  Loss: %8.3f   Acc:   %8.3f   NMI: %8.3f' % (logs.get('loss'), acc, nmi))

        # print("test: ", self.training_data.shape[0])
        # print("on epoch")
        # y_pred = self.model.predict(self.validation_data[0])
        # self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# method is KLD (KLDivergence) or KM (K means)
def cluster(  data,
            classes,
            max_epochs,
            img_size = 32,
            # outputModel  = './models/model/model',
            batch_size = 256,
            continue_training = False,
            method = "KM",
            save_dir = './results',
            # output_features_path = 'deep_autoe_features.pickle',
            # output_labels_path = 'deep_autoe_labels.pickle',
            # logs_path = '/tmp/tensorflow_logs/example/'
        ):
    # global shouldStopNow
    # signal.signal(signal.SIGINT, signal_handler)

    num_channels = data.train.num_channels
 
    # Prepare the data
    x_train = data.train.images[:,:,:,:]
    x_test = data.test.images[:,:,:,:]
    y_train = np.argmax(data.train.labels, axis=1)#np.array(classes)[np.argmax(data.train.labels, axis=1)]
    y_test = np.argmax(data.test.labels, axis=1)#np.array(classes)[np.argmax(data.test.labels, axis=1)]

     # normalize all values between 0 and 1
    x_train = x_train.astype('float32') / np.amax(x_train).astype('float32')
    x_test = x_test.astype('float32') / np.amax(x_test).astype('float32')
    x_train = np.reshape(x_train, (len(x_train), img_size, img_size, num_channels))    # adapt this if using 'channels_first' image data format
    x_test = np.reshape(x_test, (len(x_test), img_size, img_size, num_channels))       # adapt this if using 'channels_first' image data format
    print(x_train.shape)
    print(x_test.shape)


    # Prepare the convolutional deep clustering model

    # build the encoder, decoder and autoencoder models and load weights from pretraining step:
    [autoencoder, encoder, decoder] = buildAutoencoder(img_size, num_channels)
    autoencoder.load_weights(save_dir + '/ae_weights.h5')

    
    # Add the clustering layer
    n_clusters = len(np.unique(y_train))
    clustering_layer = ClusteringLayer(n_clusters, name='Clustering')(encoder.get_output_at(-1))
    model = Model(inputs=encoder.get_input_at(0), outputs=[decoder(encoder(encoder.get_input_at(0))), clustering_layer])

    # Visualize moded
    model.summary()
    plot_model(model,to_file='deepClustering.png',show_shapes=True)

    # serialize model to JSON
    model_json = model.to_json()
    with open(save_dir + "/convolutional_deep_clustering_model.json", "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")

    
    print('Data samples', x_train.shape, ", Num clusters: ", n_clusters)
    if method == 'KM':
        print("Using KMeans as Loss Function")
        unitTests()
        

        print('Initializing', n_clusters, 'cluster centers with k-means.')
        encoded_imgs = encoder.predict(x_train)
        quality_desc, cluster_centers = evaluateKMeans(encoded_imgs, y_train, n_clusters, 'Initial')
        print(quality_desc)
        # t_cluster_centers = K.variable(value=cluster_centers)
        model.get_layer("Clustering").set_weights([cluster_centers])

        # print("weights", model.get_layer("Clustering").get_weights())
        

        def clustering_loss(y_true, y_pred):
            kmeansLoss = getKMeansLoss(encoder.get_output_at(-1), y_pred, model.get_layer("Clustering").get_weights()[0], n_clusters, 128, batch_size, True)
            return kmeansLoss


        model_losses = {
            "Decoder": "binary_crossentropy",
            "Clustering": clustering_loss
        }
        lossWeights = {
            "Decoder": 1.0, 
            "Clustering": 0.1}
        
        opt = 'adam'
        model.compile(optimizer=opt, loss=model_losses, loss_weights=lossWeights)#,	metrics=["accuracy"])




        # # Print Gradient
        # listOfVariableTensors = model.trainable_weights
        # gradients = K.gradients(model.output, listOfVariableTensors)
        # print("gradients", gradients)


        print("Start Training...")
        update_interval = 10
    
        save_interval = 1
        for ite in range(int(max_epochs/update_interval)):
            print("Iteration", ite , "/", int(max_epochs/update_interval))
  
            history = model.fit(x_train, 
                        {"Decoder": x_train, "Clustering": data.train.labels},
                        validation_data=(x_test,
                            {"Decoder": x_test, "Clustering": data.test.labels}),
                        epochs=update_interval,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')], verbose=2)
            # print(history.history.keys())
            # update cluster centers

            # if ite % update_interval == 0:
            print('Updating', n_clusters, 'cluster centers with K-means.')
            Z = encoder.predict(x_train)
            quality_desc, cluster_centers = evaluateKMeans(Z, y_train, n_clusters, "%d/%d [%.4f]" % (ite + 1, ite, history.history['loss'][0]))
            print("Clustering accuracy: ", quality_desc)
            # K.set_value(t_cluster_centers, cluster_centers)
            # model.get_layer("Clustering").set_weights([cluster_centers])

            if ite % save_interval == 0:
                print('Saving model to:', save_dir + '/full_model_' + str(ite) + '.h5')
                model.save_weights(save_dir + '/full_model_' + str(ite) + '.h5')

        print('Saving model to:', save_dir + '/model_final.h5')
        model.save_weights(save_dir + '/model_final.h5')

    if method == 'KLD':
        print("Using KLDivergence as Loss Function")

        print('Initializing', n_clusters, 'cluster centers with k-means.')
        encoded_imgs = encoder.predict(x_train)
        quality_desc, cluster_centers = evaluateKMeans(encoded_imgs, y_train, n_clusters, 'Initial')
        model.get_layer("Clustering").set_weights([cluster_centers])
        # t_cluster_centers = K.variable(value=cluster_centers)

        # P is the more pure target distribution we want to achieve
        # P = K.variable(np.zeros(shape=(x_train.shape[0], n_clusters)))

        def clustering_loss(P, softAssignments):
            KLD_loss = getKLDivLossExpression(softAssignments, P)
            return KLD_loss
        
        model_losses = {
            "Decoder": "binary_crossentropy",
            "Clustering": "kld" # "Clustering": clustering_loss
        }
        lossWeights = {
            "Decoder": 1.0, 
            "Clustering": 0.1}
        
        opt = 'adadelta'
        model.compile(optimizer=opt, loss=model_losses, loss_weights=lossWeights)#,	metrics=["accuracy"])


        print("Start Training...")
        update_interval = 1
    
        def calculateP(Q):
            # Function to calculate the desired distribution Q^2, for more details refer to DEC paper
            f = Q.sum(axis=0)
            pij_numerator = Q * Q
            pij_numerator = pij_numerator / f
            normalizer_p = pij_numerator.sum(axis=1).reshape((Q.shape[0], 1))
            P = pij_numerator / normalizer_p
            return P

        # prepare callback
        clusterAccCallback = ClusterAccuracyCallback((x_train, y_train)) 

        save_interval = 10
        for ite in range(int(max_epochs/update_interval)):
            if shouldStopNow:
                
                break

            print("Iteration", ite , "/", int(max_epochs/update_interval))
  
            [_, qij] = model.predict(x_train)
            pij = target_distribution(qij)
            
            # [_, qij_test] = model.predict(x_test)
            # pij_test = target_distribution(qij_test)
            clusterAccCallback.qij = qij
            

            history = model.fit(x_train, 
                        {"Decoder": x_train, "Clustering": pij},
                        # validation_data=(x_test,
                        #     {"Decoder": x_test, "Clustering": pij_test}),
                        epochs=update_interval,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')], verbose=1)


            [_, qij] = model.predict(x_train)
            labels_pred = qij.argmax(axis=1)
            acc = np.round(cluster_acc(y_train, labels_pred), 5)
            nmi = normalized_mutual_info_score(y_train, labels_pred)
            print('\nCluster  Loss: %8.3f   Acc:   %8.3f   NMI: %8.3f' % (history.history['loss'][0], acc, nmi))

            # print(history.history.keys())
            # update cluster centers
            if ite % save_interval == 0:
                print('Saving model to:', save_dir + '/full_model_' + str(ite) + '.h5')
                model.save_weights(save_dir + '/full_model_' + str(ite) + '.h5')

        print("Losses: ", clusterAccCallback.losses)
        print("Cluster Accuracies: ", clusterAccCallback.clusterAcc)
        print('Saving model to:', save_dir + '/model_final.h5')
        model.save_weights(save_dir + '/model_final.h5')




    # Visualize some final auto encoder examples
    encoded_imgs = encoder.predict(x_train)
    decoded_imgs = decoder.predict(encoded_imgs)


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_train[i].reshape(img_size, img_size,num_channels)[:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 10, n + i+1)
        plt.imshow(decoded_imgs[i].reshape(img_size, img_size,num_channels)[:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # # display masks
        # ax = plt.subplot(3, 10, 2*n + i+1)
        # plt.imshow(x_test_mask[i].reshape(img_size, img_size))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.show()
    return


        
        


def predict(data,
            classes,
            img_size,
            n_clusters,
            batch_size,
            inputDir, 
            inputModel,
            inputWeights,
            output_features_path = 'convolutional_deep_clustering_features.pickle',
            output_labels_path = 'convolutional_deep_clustering_labels.pickle'
            ):
    
    
    num_channels = data.train.num_channels
    # load the encoder, decoder and autoencoder models:
    [autoencoder, encoder, decoder] = buildAutoencoder(img_size, num_channels)
    clustering_layer = ClusteringLayer(n_clusters, name='Clustering')(encoder.get_output_at(-1))
    model = Model(inputs=encoder.get_input_at(0), outputs=[decoder(encoder(encoder.get_input_at(0))), clustering_layer])
    model.load_weights(inputDir + inputWeights)
    model.summary()
    # from keras.models import model_from_json
    # json_file = open(inputDir + inputModel, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print("Loaded model from disk")


    x_train = data.train.images[:,:,:,:]
    x_test = data.test.images[:,:,:,:]
    y_train = np.array(classes)[np.argmax(data.train.labels, axis=1)]
    y_test = np.array(classes)[np.argmax(data.test.labels, axis=1)]
     # normalize all values between 0 and 1
    x_train = x_train.astype('float32') / np.amax(x_train).astype('float32')
    x_test = x_test.astype('float32') / np.amax(x_test).astype('float32')
    x_train = np.reshape(x_train, (len(x_train), img_size, img_size, num_channels))    # adapt this if using 'channels_first' image data format
    x_test = np.reshape(x_test, (len(x_test), img_size, img_size, num_channels))       # adapt this if using 'channels_first' image data format


    encoded_imgs = encoder.predict(x_train)
    decoded_imgs = decoder.predict(encoded_imgs)

    pickle.dump(encoded_imgs, open(output_features_path, 'wb'))
    pickle.dump(y_train, open(output_labels_path, 'wb'))


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_train[i].reshape(img_size, img_size,num_channels)[:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 10, n + i+1)
        plt.imshow(decoded_imgs[i].reshape(img_size, img_size,num_channels)[:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # # display masks
        # ax = plt.subplot(3, 10, 2*n + i+1)
        # plt.imshow(x_test_mask[i].reshape(img_size, img_size))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    plt.show()
    return