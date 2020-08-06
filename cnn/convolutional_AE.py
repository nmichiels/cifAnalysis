# Based on: https://github.com/snatch59/keras-autoencoders

from cifDataset.cifStreamer.dataset import Dataset
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Reshape,Concatenate,BatchNormalization,Activation,Dropout,LeakyReLU,AvgPool2D
from keras.models import Model,Sequential
from keras import backend as K
from keras import regularizers
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import pickle

def Encoder(input_img):
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return Model(input_img, encoded, name="Encoder")



def Decoder_32(input_img):  
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded, name="Decoder")


def Decoder_lessThan32(input_img):  
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded, name="Decoder")


def Encoder_40_v0(input_img, nchannels):  

    # Encoder Layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # Flatten encoding for visualization
    #x = Flatten(name="encoded_feature_vector")(x)
    #encoded = Reshape((5, 5, 8))(x)
    encoded = x
    return Model(input_img, encoded, name="Encoder")




def Decoder_40_v0(input_img, nchannels):  
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(nchannels, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded, name="Decoder") 
    return decoder



def Encoder_40_v1(input_img, nchannels):  
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Flatten encoding for visualization
    x = Flatten(name="latent_space")(x)
    encoded = Reshape((5, 5, 16))(x)
    return Model(input_img, encoded, name="Encoder")





def Decoder_40_v1(input_img, nchannels):  
    x = Conv2D(16, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(nchannels, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return Model(input_img, decoded, name="Decoder") 
    return decoder



def Encoder_40_v2(input_img, nchannels):  
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Flatten encoding for visualization
    x = Flatten(name="latent_space")(x)
    encoded = Reshape((5, 5, 16))(x)
    return Model(input_img, encoded, name="Encoder")





def Decoder_40_v2(input_img, nchannels):  
    x = Conv2D(16, (3, 3), padding='same')(input_img)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(nchannels, (3, 3), padding='same')(x)
    #x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return Model(input_img, decoded, name="Decoder") 
    return decoder




def Encoder_40_v3(input_img, nchannels):  
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Flatten encoding for visualization
    x = Flatten(name="latent_space")(x)
    encoded = Reshape((5, 5, 32))(x)
    return Model(input_img, encoded, name="Encoder")





def Decoder_40_v3(input_img, nchannels):  
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(nchannels, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    return Model(input_img, decoded, name="Decoder") 
    return decoder

def predict(    data, 
                classes, 
                channels, 
                img_size,
                inputModel,
                batch_size = 256,
                output_features_path = 'convolutional_autoe_features.pickle',
                output_labels_path = 'convolutional_autoe_labels.pickle',
                log_dir = './tmp/tensorflow_logs/example/'
            ):
    from keras.models import model_from_json
    from os import makedirs
    from os.path import exists, join

    num_channels = data.test.num_channels
    #input_img = Input(shape=(img_size, img_size, num_channels))


    # load json and create model
    json_file_ae = open(inputModel+"/model_ae.json", 'r')
    json_string_ae = json_file_ae.read()
    json_file_ae.close()
    autoencoder = model_from_json(json_string_ae)
    # load weights into autoencoder
    autoencoder.load_weights(inputModel + '/ae_weights.h5')
    print("Loaded model from disk")
    

    encoder = autoencoder.get_layer('Encoder')
    encoder.summary()
    
    decoder = autoencoder.get_layer('Decoder')
    decoder.summary()

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.summary()



    x_test = data.test.images[:,:,:,:]
    y_test = np.array(classes)[np.argmax(data.test.labels, axis=1)]
    print(y_test)

    # normalize all values between 0 and 1
    x_test = x_test.astype('float32') / np.amax(x_test).astype('float32')
    x_test = np.reshape(x_test, (len(x_test), img_size, img_size, num_channels))       # adapt this if using 'channels_first' image data format
    
    print("Test dataset size:", x_test.shape)

    print(y_test)
    if not exists(log_dir):
        makedirs(log_dir)

     # save class labels to disk to color data points in TensorBoard accordingly
    with open(join(log_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, y_test, fmt='%s', delimiter='\n')

    # take a look at the reconstructed digits
    # decoded_imgs = autoencoder.predict(x_test)
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    pickle.dump(encoded_imgs, open(output_features_path, 'wb'))
    pickle.dump(y_test, open(output_labels_path, 'wb'))

    
   

    with open(join(log_dir, 'features.tsv'), 'w') as f:
        np.savetxt(f, np.reshape(encoded_imgs, (encoded_imgs.shape[0], -1)), delimiter='\t')
    
def embedding(encoded_imgs, encoded_labels, logdir, name):
    num_of_samples=feature_vectors.shape[0]

    x, y = data.test.next_batch(num_of_samples)

    y_idx = np.argmax(y, axis=1)
    y = [classes[classIdx] for classIdx in y_idx]
    y = np.array(y)


    
    features = tf.Variable(feature_vectors, name='features')

    metadata_file = open(os.path.join(logdir, 'metadata_' + name + '.tsv'), 'w')
    metadata_file.write('Class\tName\n')
    for i in range(num_of_samples):
        metadata_file.write('{}\t{}\n'.format(y_idx[i],y[i]))
    metadata_file.close()

    with tf.Session() as sess:
        saver = tf.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, os.path.join(logdir, 'images_' + name + '.kpt'))
        
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata_' + name + '.tsv'#os.path.join(logdir, 'metadata_' + name + '.tsv')
        # Comment out if you don't want sprites
        # embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
        # embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(logdir), config)





def train(  data,
            classes,
            max_epochs,
            img_size = 40,
            modelNr = 1, # defines the complexity of the AE model, with larger numbers going towards more complexity 
            outputModel  = './models/model/model',
            batch_size = 256,
            #output_features_path = 'convolutional_autoe_features.pickle',
            #output_labels_path = 'convolutional_autoe_labels.pickle',
            #log_dir = '/tmp/tensorflow_logs/example/'
        ):
    from keras.models import model_from_json
    from os import makedirs
    from os.path import exists, join

    num_channels = data.train.num_channels

    # define input to the encoder and decoder model:
    input_img = Input(shape=(img_size, img_size, num_channels))



    if (img_size == 40):
        encoder_name = 'Encoder_40_v' + repr(modelNr)
        encoder = eval(encoder_name)(input_img, num_channels)
        if modelNr == 0:
            encoded_img = Input(shape=(5, 5, 8))
        elif modelNr == 1:
            encoded_img = Input(shape=(5, 5, 16))
        elif modelNr == 2:
            encoded_img = Input(shape=(5, 5, 16))
        elif modelNr == 3:
            encoded_img = Input(shape=(5, 5, 32))
        else:
            print("Error unknown encoder model number", modelNr)
    else:
        encoded_img = Input(shape=(4, 4, 8))
        encoder = Encoder(input_img)
    encoder.summary()
    # make the encoder, decoder and autoencoder models:
    

    


    if (img_size > 32):
        decoder_name = 'Decoder_40_v' + repr(modelNr)
        decoder = eval(decoder_name)(encoded_img, num_channels)
        #if modelNr == 0:
        #    decoder = Decoder_40_v1(encoded_img, num_channels)
        #elif modelNr == 1:
        #    decoder = Decoder_40_v2(encoded_img, num_channels)
        #else:
        #    print("Error unknown decoder model number", modelNr)
    elif (img_size == 32):
        decoder = Decoder_32(encoded_img)
    else:
        decoder = Decoder_lessThan32(encoded_img)
    decoder.summary()




    autoencoder = Model(input_img, decoder(encoder(input_img)), name="Autoencoder")

 

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')#, metrics=['accuracy'])


    if not exists(outputModel):
        makedirs(outputModel)
    # checkpoint
    filepath=outputModel + "/weights-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto ')

    json_string_encoder = encoder.to_json()
    json_string_decoder = decoder.to_json()
    json_string_ae = autoencoder.to_json()
    with open(outputModel+"/model_encoder.json", "w") as json_file:
        json_file.write(json_string_encoder)
    with open(outputModel+"/model_decoder.json", "w") as json_file:
        json_file.write(json_string_decoder)
    with open(outputModel+"/model_ae.json", "w") as json_file:
        json_file.write(json_string_ae)


    autoencoder.summary()
  

    x_train, y_train = data.train.nextBatch(data.train.num_examples, img_size)
    x_test, y_test = data.test.nextBatch(data.test.num_examples, img_size)


     # normalize all values between 0 and 1
    x_train = x_train.astype('float32') / np.amax(x_train).astype('float32')
    x_test = x_test.astype('float32') / np.amax(x_test).astype('float32')
    x_train = np.reshape(x_train, (len(x_train), img_size, img_size, num_channels))    # adapt this if using 'channels_first' image data format
    x_test = np.reshape(x_test, (len(x_test), img_size, img_size, num_channels))       # adapt this if using 'channels_first' image data format
    print(x_train.shape)
    print(x_test.shape)

 

    from keras.callbacks import TensorBoard


    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_test[i].reshape(img_size, img_size,num_channels)[:,:,0])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, 10, n + i+1)
        plt.imshow(x_test[i].reshape(img_size, img_size,num_channels)[:,:,0])
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


    #if not exists(log_dir):
    #    makedirs(log_dir)
    ## save class labels to disk to color data points in TensorBoard accordingly
    #with open(join(log_dir, 'metadata.tsv'), 'w') as f:
    #    np.savetxt(f, np.argmax(data.test.labels, axis=1))

    #tensorboard = TensorBoard(batch_size=batch_size,
    #                      embeddings_freq=1,
    #                      embeddings_layer_names=['latent_space'],
    #                      embeddings_metadata='metadata.tsv',
    #                      embeddings_data=x_test)
     
    autoencoder.fit(x_train, x_train,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[checkpoint], verbose=2)

    print("Saving model weights to:")
    print(outputModel + '/encoder_weights.h5')
    print(outputModel + '/decoder_weights.h5')
    encoder.save_weights(outputModel + '/encoder_weights.h5')
    decoder.save_weights(outputModel + '/decoder_weights.h5')
    autoencoder.save_weights(outputModel + '/ae_weights.h5')
    

    # take a look at the reconstructed digits
    # decoded_imgs = autoencoder.predict(x_test)
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)



    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_test[i].reshape(img_size, img_size,num_channels)[:,:,0])
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







def train_gridSearchCV(  data,
            classes,
            channels,
            max_epochs,
            img_size = 40,
            modelNr = 1, # defines the complexity of the AE model, with larger numbers going towards more complexity 
            outputModel  = './models/model/model',
            batch_size = 256,
            #output_features_path = 'convolutional_autoe_features.pickle',
            #output_labels_path = 'convolutional_autoe_labels.pickle',
            #log_dir = '/tmp/tensorflow_logs/example/'
        ):
    from keras.models import model_from_json
    from os import makedirs
    from os.path import exists, join
    from keras.callbacks import ModelCheckpoint
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV

    num_channels = len(channels)#data.train.num_channels


    def make_model(ae_layer_sizes, poolingType, doubleLayers, kernel_size, dropout):


       
        # define input to the encoder and decoder model:
        inputs  = Input(shape=(img_size, img_size, num_channels))
        activation = LeakyReLU()

 
        encoded = inputs#Conv2D(depth, 1, padding='same')(inputs)
        for layer_size in ae_layer_sizes:
         
            encoded = Conv2D(layer_size, kernel_size, padding='same')(encoded)
            encoded = LeakyReLU()(encoded)
            if doubleLayers:
                encoded = Conv2D(layer_size, kernel_size, padding='same')(encoded)
                encoded = LeakyReLU()(encoded)
            if poolingType == 'AvgPool2D':
                encoded = AvgPool2D(2)(encoded)
            else:
                encoded = MaxPooling2D(2)(encoded)
        
            encoded = Dropout(dropout)(encoded)
        
        encoder = Model(inputs, encoded)
        latentShape = encoder.layers[len(encoder.layers)-1].output_shape

     
        #encoder.summary()
        
        

        inputLatent = Input(shape=(latentShape[1],latentShape[2],latentShape[3]))
        #print(inputLatent)
        decoded = inputLatent#encoded
        for i in reversed(range(len(ae_layer_sizes))):
            decoded = Conv2D(ae_layer_sizes[i], kernel_size, padding='same')(decoded)
            decoded = LeakyReLU()(decoded)
            if doubleLayers:
                decoded = Conv2D(ae_layer_sizes[i], kernel_size, padding='same')(decoded)
                decoded = LeakyReLU()(decoded)
            decoded = UpSampling2D(2)(decoded)
            decoded = Dropout(dropout)(decoded)

        decoded = Conv2D(num_channels, kernel_size, padding='same', name='decoded')(decoded)
        decoded = Activation('sigmoid')(decoded)
  


        decoder = Model(inputLatent, decoded)
        #decoder.summary()



        autoencoder =  Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
        autoencoder.summary()

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

       
        return encoder, decoder, autoencoder
      
    
    encoder, decoder, autoencoder = make_model([64,32,32], 'MaxPool2D', False, 3, 0.25)
    autoencoder.summary()

    x_train, y_train = data.train.next_batch(data.train.num_examples, img_size)
    x_train = x_train[:,:,:,channels]

    x_valid, y_valid = data.test.next_batch(data.test.num_examples, img_size)
    x_valid = x_valid[:,:,:,channels]

    filepath= outputModel + "/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint]

    autoencoder.fit(x_train, x_train,
                batch_size=batch_size,
                epochs=100,
                callbacks=callbacks_list,
                verbose=1,
                validation_data=(x_valid, x_valid),
                shuffle=True)



    print("Saving model weights to:")
    print(outputModel + '/encoder_weights.h5')
    print(outputModel + '/decoder_weights.h5')
    encoder.save_weights(outputModel + '/encoder_weights.h5')
    decoder.save_weights(outputModel + '/decoder_weights.h5')
    autoencoder.save_weights(outputModel + '/ae_weights.h5')
    

    # take a look at the reconstructed digits
    # decoded_imgs = autoencoder.predict(x_test)
    encoded_imgs = encoder.predict(x_valid)
    decoded_imgs = decoder.predict(encoded_imgs)



    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, 10, i+1)
        plt.imshow(x_valid[i].reshape(img_size, img_size,num_channels)[:,:,0])
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



