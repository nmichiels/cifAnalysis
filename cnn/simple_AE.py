# Based on: https://github.com/snatch59/keras-autoencoders

from dataset.dataset import DataSet
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train(  data,
            classes,
            max_epochs,
            img_size = 40,
            outputModel  = './models/model/model',
            batch_size = 256,
            output_features_path = 'simple_autoe_features.pickle',
            output_labels_path = 'simple_autoe_labels.pickle',
            logs_path = '/tmp/tensorflow_logs/example/'
        ):
    
    num_channels = data.train.num_channels


    use_regularizer = True
    my_regularizer = None
    if use_regularizer:
        # add a sparsity constraint on the encoded representations
        # note use of 10e-5 leads to blurred results
        my_regularizer = regularizers.l1(10e-8)
        # and a larger number of epochs as the added regularization the model
        # is less likely to overfit and can be trained longer
        # features_path = 'sparse_autoe_features.pickle'
        # labels_path = 'sparse_autoe_labels.pickle'




        # this is the size of our encoded representations
    encoding_dim = 32   # 32 floats -> compression factor 24.5, assuming the input is 784 floats

    # this is our input placeholder; 784 = 28 x 28
    input_img = Input(shape=(img_size*img_size*num_channels, ))

    # "encoded" is the encoded representation of the inputs
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(img_size*img_size*num_channels, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # Separate Encoder model

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # Separate Decoder model

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))


    # configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


    x_train = data.train.images[:,:,:,:]
    x_test = data.test.images[:,:,:,:]

    # y_train = np.array(classes)[np.argmax(data.train.labels, axis=1)]
    y_test = np.array(classes)[np.argmax(data.test.labels, axis=1)]

    print("y_test size: ", y_test.shape)

    # normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
    print("np.amax(x_train)", np.amax(x_train))
    x_train = x_train.astype('float32') / np.amax(x_train).astype('float32')
    x_test = x_test.astype('float32') / np.amax(x_test).astype('float32')
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)



    from keras.callbacks import TensorBoard


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_test[i].reshape(img_size, img_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 10, n + i+1)
        plt.imshow(x_test[i].reshape(img_size, img_size))
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




    autoencoder.fit(x_train, x_train,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')], verbose=2)



    # decoded_imgs = autoencoder.predict(x_test)

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)


    # save latent space features 32-d vector
    pickle.dump(encoded_imgs, open(output_features_path, 'wb'))
    pickle.dump(y_test, open(output_labels_path, 'wb'))


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_test[i].reshape(img_size, img_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 10, n + i+1)
        plt.imshow(decoded_imgs[i].reshape(img_size, img_size))
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



