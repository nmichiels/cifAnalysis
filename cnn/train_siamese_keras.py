from cifDataset.cifStreamer.dataset import Dataset
import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer
import time
class MyLayer(Layer):

    # initialize the layer, and set an extra parameter axis. No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(MyLayer, self).__init__(**kwargs)

    # first use build function to define parameters, Creates the layer weights.
    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        print("test input", input_shape)
        super(MyLayer, self).build(input_shape)

    # This is where the layer's logic lives. In this example, I just concat two tensors.
    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        self.result = K.sqrt(K.maximum(sum_square, K.epsilon()))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        #print("layer shape:", K.int_shape(self.result))
        return K.int_shape(self.result)


def train(  data,
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

    num_classes = data.train.num_classes
    num_channels = len(channels)

    import random
    from keras.datasets import mnist
    from keras.models import Model
    from keras.layers import Input, Flatten, Dense, Dropout, Lambda
    from keras.optimizers import RMSprop
    
    from keras.callbacks import ModelCheckpoint
    import os
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)


    def euclidean_distance(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


    def create_pairs(x, digit_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        print(len(pairs))
        print(len(labels))
        return np.array(pairs), np.array(labels)

    def make_model(input_shape, conv_layer_sizes, dense_layer_sizes, kernel_size, pool_size,dropout):
        '''Creates model comprised of 2 convolutional layers followed by dense layers
        dense_layer_sizes: List of layer sizes.
            This list has one number for each layer
        filters: Number of convolutional filters in each convolutional layer
        kernel_size: Convolutional kernel size
        pool_size: Size of pooling area for max pooling
        '''
        from keras.models import Sequential, Model
        from keras.layers import Input, Dense, Conv2D, Flatten,Dropout,MaxPooling2D,LeakyReLU,Activation

        model = Sequential()
        # define input to the encoder and decoder model:
        inputs  = Input(shape=input_shape)
        activation = LeakyReLU()

 
        cnn = inputs#Conv2D(depth, 1, padding='same')(inputs)
        for layer_size in conv_layer_sizes:
         
            cnn = Conv2D(layer_size, kernel_size, padding='same')(cnn)
            cnn = LeakyReLU()(cnn)
            cnn = MaxPooling2D(pool_size)(cnn)
            cnn = Dropout(dropout)(cnn)

        cnn = Flatten(name='activations')(cnn)
        for layer_size in dense_layer_sizes:
            cnn = Dense(layer_size)(cnn)
            cnn = LeakyReLU()(cnn)

        cnn = Dropout(dropout)(cnn)
        cnn = Dense(num_classes)(cnn)
        cnn = Activation('softmax')(cnn)

        model = Model(inputs, cnn)

        # model.compile(loss=keras.losses.categorical_crossentropy,
        #                 optimizer=keras.optimizers.Adadelta(),
        #               metrics=['accuracy'])
       

        return model 

    def create_base_network(input_shape):
        '''Base network to be shared (eq. to feature extraction).
        '''
        input = Input(shape=input_shape)
        x = Flatten()(input)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        return Model(input, x)


    def compute_accuracy(y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        pred = y_pred.ravel() < 0.5
        return np.mean(pred == y_true)


    def accuracy(y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))



    x_train, y_train = data.train.nextBatch(data.train.num_examples, img_size)
    x_train = x_train[:,:,:,channels]

    x_test, y_test = data.test.nextBatch(data.test.num_examples, img_size)
    x_test = x_test[:,:,:,channels]

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    

    # # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    input_shape = x_train.shape[1:]

    print(np.where(y_train == 1)[0])
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    # network definition
    # base_network = create_base_network(input_shape)
    base_network = make_model(input_shape, conv_layer_sizes=[32,64], dense_layer_sizes=[64], kernel_size=5, pool_size=2,dropout=0.5)
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)


    # distance = Lambda(lambda t: euclidean_distance(t))([processed_a, processed_b]) # custom layer because lambda layer cannot be saved to json

    distance = MyLayer()([processed_a, processed_b])
    # distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b]) # original
    # print("out shape: ", eucl_dist_output_shape())
    model = Model([input_a, input_b], distance)
    model.summary()

    import os
    # define safe directory
    outputDir = os.path.dirname(outputDir) ## directory of output model
    saveDir = os.path.join(outputDir,outputModel)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # serialize model to JSON
    print('Saving model to:', saveDir + '/model.json')
    model_json = model.to_json()
    with open(saveDir + "/model.json", "w") as json_file:
        json_file.write(model_json)


    # train
    rms = RMSprop()
    # model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    model.compile(loss=contrastive_loss,
                        optimizer=rms,
                      metrics=[accuracy])#, 'accuracy'])
   


    # checkpoint
    filepath= saveDir + "/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=True,mode='max', period=1)
    callbacks_list = [checkpoint]

    

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
            batch_size=128,
            epochs=max_epochs,
            callbacks=callbacks_list,
            validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    # serialize weights to HDF5
    model.save_weights(saveDir + "/final_weights.hdf5")
     
    # from vis.visualization import visualize_saliency
    # from vis.utils import utils
    # idx = 0
    # from keras.utils import CustomObjectScope
    # with CustomObjectScope({'MyLayer': MyLayer,'accuracy': accuracy, 'contrastive_loss' : contrastive_loss}): 
    #     grads = visualize_saliency(model, -2, filter_indices=tr_y[idx], 
    #                                         seed_input=[tr_pairs[idx, 0], tr_pairs[idx, 1]], backprop_modifier='guided', keepdims=True)
    # print("output", grads.shape)


    # LOSS Learning curves
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, (len(history.history['val_accuracy']) + 1))
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # ACCURACY Learning Curves

    history_dict = history.history
    loss_values = history_dict['accuracy']
    val_loss_values = history_dict['val_accuracy']
    epochs = range(1, (len(history.history['accuracy']) + 1))
    plt.plot(epochs, loss_values, 'bo', label='Training Acc')
    plt.plot(epochs, val_loss_values, 'b', label='Validation Acc')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    # Plot the representation learned from the siamese network
    #embedding_model = model.layers[2]
    #embeddings = embedding_model.predict(x_train)
    embeddings = base_network.predict(x_train)

    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=2,random_state=10).fit_transform(embeddings)

    mnist_classes = ['0', '1']#, '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['#1f77b4', '#ff7f0e']#, '#2ca02c', '#d62728',
               # '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                #'#bcbd22', '#17becf']

    plt.figure(figsize=(10,10))
    for i in range(2):
        inds = np.where(y_train==i)[0]
        plt.scatter(X_embedded[inds,0], X_embedded[inds,1], alpha=0.5, color=colors[i])
    plt.legend(mnist_classes)
    plt.show()