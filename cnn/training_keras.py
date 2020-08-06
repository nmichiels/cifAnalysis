from cifDataset.cifStreamer.dataset import Dataset
import numpy as np
import keras
import time
from .keras_utils import My_Generator, My_GeneratorIncremental


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
            augmentation = False,
            addGaussianNoise = False,
            masking = False,
            residual = False,
            validation_size = 0.2,
            display_epoch = 1,
            dropout = 0.25,
            initialWeights = None,
            logs_path = './tmp/tensorflow_logs/example/'
        ):
    
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Conv2D, Flatten,Dropout,MaxPooling2D,LeakyReLU,Activation,LocallyConnected2D,GaussianNoise
    from keras.callbacks import ModelCheckpoint
    from colorama import Fore, Back, Style
    import os
    from numpy.random import seed
    seed(1)
    import tensorflow
    #tensorflow.random.set_seed(2)


    # from tensorflow import set_random_seed
    # set_random_seed(2)

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


 
    #Network graph params
    filter_size_conv1 = 3 
    num_filters_conv1 = 32
    filter_size_conv2 = 3
    num_filters_conv2 = 32
    filter_size_conv3 = 3
    num_filters_conv3 = 64
    fc_layer_size = 128

    def make_model(conv_layer_sizes, dense_layer_sizes, kernel_size, pool_size,dropout, withNoise = False):
        '''Creates model comprised of 2 convolutional layers followed by dense layers
        dense_layer_sizes: List of layer sizes.
            This list has one number for each layer
        filters: Number of convolutional filters in each convolutional layer
        kernel_size: Convolutional kernel size
        pool_size: Size of pooling area for max pooling
        '''

        model = Sequential()
        # define input to the encoder and decoder model:
        inputs  = Input(shape=(img_size, img_size, num_channels))
        activation = LeakyReLU()

 
        cnn = inputs#Conv2D(depth, 1, padding='same')(inputs)
        if withNoise:
            cnn = GaussianNoise(0.1)(cnn)
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
    
        from tensorflow.python.keras.optimizer_v2.adam import Adam
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer='adam',#opt,#'adam',
                    #    optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), #keras.optimizers.Adadelta(),
                    #  optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
       

        return model 

    def make_model_locally_connected(conv_layer_sizes, dense_layer_sizes, kernel_size, pool_size,dropout):
        '''Creates model comprised of 2 convolutional layers followed by dense layers
        dense_layer_sizes: List of layer sizes.
            This list has one number for each layer
        filters: Number of convolutional filters in each convolutional layer
        kernel_size: Convolutional kernel size
        pool_size: Size of pooling area for max pooling
        '''

        model = Sequential()
        # define input to the encoder and decoder model:
        inputs  = Input(shape=(img_size, img_size, num_channels))
        activation = LeakyReLU()

 
        cnn = inputs#Conv2D(depth, 1, padding='same')(inputs)
        for counter, layer_size in enumerate(conv_layer_sizes):
            if counter is 0:
                cnn = Conv2D(layer_size, kernel_size, padding='same')(cnn)
            else:
                cnn = LocallyConnected2D(layer_size, kernel_size, padding='valid')(cnn)
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

        model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer='adadelta',#keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
       

        return model 

    # create model
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size,img_size,num_channels)))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # # model.add(Conv2D(num_filters_conv3, kernel_size=filter_size_conv3, activation='relu'))
    # model.add(Dropout(rate = 0.25))
    # model.add(Flatten(name='activations'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(rate = 0.5))
    # model.add(Dense(num_classes, activation='softmax'))


    # make best model according to gridsearchCV
    if residual:
        from cnn.keras_residualmodel import make_residual_model
        model = make_residual_model(img_size, num_channels, num_classes)
        pass
    else:
        model = make_model(conv_layer_sizes=[32,64], dense_layer_sizes=[64], kernel_size=5, pool_size=2,dropout=0.45, withNoise=addGaussianNoise )
        # model = make_model_locally_connected(conv_layer_sizes=[32,64], dense_layer_sizes=[64], kernel_size=5, pool_size=2,dropout=0.5)
        
    model.summary()

    if initialWeights is not None:
        # load initial weights into new model, used for retraining
        model.load_weights(initialWeights)


    # model.compile(loss=keras.losses.categorical_crossentropy,
    #           optimizer=keras.optimizers.Adadelta(),
    #           metrics=['accuracy'])

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


    # checkpoint
    filepath= saveDir + "/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max', period=1)
    callbacks_list = [checkpoint]


    useGenerator = True
    valid_loss = 0
    valid_acc = 0
    y_pred = None
    history = None
    if useGenerator:
        my_training_batch_generator = My_Generator(data.train, batch_size, img_size, channels)
        my_validation_batch_generator = My_Generator(data.test, batch_size, img_size, channels)

        history = model.fit_generator(generator=my_training_batch_generator,
                            steps_per_epoch=(data.train.num_examples // batch_size + 1),
                            epochs=max_epochs,
                            callbacks=callbacks_list,
                            verbose=1,
                            validation_data=my_validation_batch_generator,
                            validation_steps=(data.test.num_examples // batch_size ),
                            use_multiprocessing=False,
                            workers=1,
                            max_queue_size=1,
                            shuffle=True)
        

        valid_loss, valid_acc =  model.evaluate_generator(generator=my_validation_batch_generator, 
                                                        steps=(data.test.num_examples // batch_size + 1), 
                                                        max_queue_size = 32,
                                                        use_multiprocessing=False,
                                                        workers = 8)
        


    
    

        # Predicting the Test set results
        y_pred = model.predict_generator(generator=my_validation_batch_generator, 
                                                        steps=(data.test.num_examples // batch_size + 1), 
                                                        max_queue_size = 32,
                                                        use_multiprocessing=False,
                                                        workers = 8)
        print("Prediction shape", y_pred.shape)
    else:
        
        # Load all the data in array
        x_train, y_train = data.train.nextBatch(data.train.num_examples, img_size)
        x_train = x_train[:,:,:,channels]

        x_valid, y_valid = data.test.nextBatch(data.test.num_examples, img_size)
        x_valid = x_valid[:,:,:,channels]

        # # global standardization
        # stds = x_train.std(axis=(0,1,2), dtype='float64')
        # means = x_train.mean(axis=(0,1,2), dtype='float64')
        # for i, img in enumerate(x_train):
        #     for c in range(x_train.shape[-1]):
        #         # print(c)
        #         x_train[i,:,:,c] = (img[:,:,c] - means[c]) / stds[c]
        # stds = x_valid.std(axis=(0,1,2), dtype='float64')
        # means = x_valid.mean(axis=(0,1,2), dtype='float64')
        # for i, img in enumerate(x_valid):
        #     for c in range(x_valid.shape[-1]):
        #         # print(c)
        #         x_valid[i,:,:,c] = (img[:,:,c] - means[c]) / stds[c]


        
        # # Local Standardization
        # # calculate per-channel means and standard deviations
        # for i, img in enumerate(x_train):
        #     # print("img:" , img.shape)
        #     means = img.mean(axis=(0,1), dtype='float64')
        #     stds = img.std(axis=(0,1), dtype='float64')
        #     # print('Means: %s, Stds: %s' % (means, stds))
        #     # per-channel standardization of pixels
        #     x_train[i,:,:,:] = (img - means) / stds
        #     # confirm it had the desired effect
        #     # means = img.mean(axis=(0,1), dtype='float64')
        #     # stds = img.std(axis=(0,1), dtype='float64')
        #     # print('Means: %s, Stds: %s' % (means, stds))

        # for i, img in enumerate(x_valid):
        #     # print("img:" , img.shape)
        #     means = img.mean(axis=(0,1), dtype='float64')
        #     stds = img.std(axis=(0,1), dtype='float64')
        #     # print('Means: %s, Stds: %s' % (means, stds))
        #     # per-channel standardization of pixels
        #     x_valid[i,:,:,:] = (img - means) / stds
        #     # confirm it had the desired effect
        #     # means = img.mean(axis=(0,1), dtype='float64')
        #     # stds = img.std(axis=(0,1), dtype='float64')
        #     # print('Means: %s, Stds: %s' % (means, stds))


        # doubleData = False
        # for slice1 in range(x_train.shape[0]):
        #     if slice1%10 is 0:
        #         print(slice1)
        #     for slice2 in reversed(range(x_valid.shape[0])):
        #         theSame = np.array_equal(x_train[slice1,:,:,:], x_valid[slice2,:,:,:])
        #         if theSame is True:
        #             print(slice1, "==", slice2)
        #             print(x_train[slice1,:,:,:])
        #             print("\n")
        #             print(x_valid[slice2,:,:,:]  )
        #         doubleData = doubleData or theSame
        # print("Double data: ", doubleData)

        if residual:
            y_train = np.argmax(y_train, axis=1)
            y_valid = np.argmax(y_valid, axis=1)

        if augmentation:
            from keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.10, # Randomly zoom image 
                width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
                brightness_range=[0.75,1.25],
                horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images
            datagen.fit(x_train)

            
            validation_generator = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False  # divide each input by its std
            )  
            validation_generator.fit(x_valid)


        


            # fits the model on batches with real-time data augmentation:
            history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=len(x_train) // batch_size, epochs=max_epochs, callbacks=callbacks_list,verbose=1,
                    validation_data=(x_valid, y_valid),
                    #validation_data=validation_generator.flow(x_valid, y_valid, batch_size=batch_size),
                    #validation_steps = len(x_valid) // batch_size,
                    shuffle=True)
        else:
            print(x_train.shape)
            print(y_train.shape)
            print(callbacks_list)
            print(x_valid.shape)
            print(y_valid.shape)
            print(batch_size)
            history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=max_epochs,
                    callbacks=callbacks_list,
                    verbose=1,
                    validation_data=(x_valid, y_valid),
                    shuffle=True)

        valid_loss, valid_acc = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)

        y_pred = model.predict(x=x_valid, batch_size=batch_size)

        

    # serialize weights to HDF5
    model.save_weights(saveDir + "/final_weights.hdf5")

    history_dict = history.history

    # LOSS Learning curves
    np.savetxt(saveDir + '/history_val_los.csv', np.asarray(history_dict['val_loss']))
    np.savetxt(saveDir + '/history_val_acc.csv', np.asarray(history_dict['val_acc']))
    np.savetxt(saveDir + '/history_train_los.csv', np.asarray(history_dict['loss']))
    np.savetxt(saveDir + '/history_train_acc.csv', np.asarray(history_dict['acc']))

    

    
    # import matplotlib.pyplot as plt
    
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']
    # epochs = range(1, (len(history.history['val_acc']) + 1))
    # plt.plot(epochs, loss_values, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.ylim(0.0,0.5)
    # plt.legend()
    # plt.savefig(saveDir + '/loss.png', bbox_inches='tight')
    # # plt.show()
    # plt.clf()

    # # ACCURACY Learning Curves

    # history_dict = history.history
    # loss_values = history_dict['acc']
    # val_loss_values = history_dict['val_acc']
    # epochs = range(1, (len(history.history['acc']) + 1))
    # plt.plot(epochs, loss_values, 'bo', label='Training Acc')
    # plt.plot(epochs, val_loss_values, 'b', label='Validation Acc')
    # plt.title('Training and validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.ylim(0.775,1.0)
    # plt.legend()
    # plt.savefig(saveDir + '/acc.png', bbox_inches='tight')
    # # plt.show()

    


    print("Keras Optimization Finished with in ", max_epochs, " epochs, with", valid_acc , "validation accuracy.")




class EstimatorSelectionHelper:
    

    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.grid_search = 0

    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        from sklearn.model_selection import GridSearchCV
        print("Running GridSearchCV")
        self.grid_search = GridSearchCV(self.model, self.params, cv=cv, n_jobs=n_jobs, 
                            verbose=verbose, scoring=scoring, refit=refit)
        self.grid_search.fit(X,y)  

    def score_summary(self, sort_by='mean_score'):
        import pandas as pd
        res = self.grid_search.cv_results_
        return pd.DataFrame.from_dict(res)

    def best_params(self):
        return self.grid_search.best_params_

    def best_model(self):
        return self.grid_search.best_estimator_.model

    

def train_gridSearchCV(  data,
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
            dropout = 0.125,
            logs_path = './tmp/tensorflow_logs/example/'
        ):
    
    from keras.models import Sequential,Model
    from keras.layers import Input, Dense, Conv2D, Flatten,Dropout,MaxPooling2D,Activation, LeakyReLU
    from keras.callbacks import ModelCheckpoint
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    

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
        if i < len(counts):
            print(className, ":" , counts[i], end =" ")
        else:
            print(className, ":" , 0, end =" ")
    print("]")
    print("Number of channels to train on:\t\t\t%6d"%(len(channels)), channels, "\n")
    print(Style.RESET_ALL)

    num_classes = data.train.num_classes
    num_channels = len(channels)#data.train.num_channels

    input_shape=(img_size,img_size,num_channels)
    
       


    def make_model(conv_layer_sizes, dense_layer_sizes, kernel_size, pool_size,dropout, lr):
        '''Creates model comprised of 2 convolutional layers followed by dense layers
        dense_layer_sizes: List of layer sizes.
            This list has one number for each layer
        filters: Number of convolutional filters in each convolutional layer
        kernel_size: Convolutional kernel size
        pool_size: Size of pooling area for max pooling
        '''

        model = Sequential()
        # define input to the encoder and decoder model:
        inputs  = Input(shape=(img_size, img_size, num_channels))
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

        # opt = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',#opt,#,'adadelta',
                      metrics=['accuracy'])

        return model   

    # define safe directory
    outputDir = os.path.dirname(outputDir) ## directory of output model
    saveDir = os.path.join(outputDir,outputModel)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)


    x_train, y_train = data.train.nextBatch(data.train.num_examples, img_size)
    y_train = y_train.argmax(axis=1)
    

    x_train = x_train[:,:,:,channels]

    #x_valid, y_valid = data.test.next_batch(data.test.num_examples, img_size)
    #x_valid = x_valid[:,:,:,channels]

 


    #x_train = np.concatenate((x_train,x_valid))
    #y_train = np.concatenate((y_train,y_valid))
  

  
    


    my_classifier = KerasClassifier(make_model, batch_size=batch_size)
    scoring = {'accuracy', 'f1_micro'}



    # # grid search !
    # conv_size_candidates = [[32,64],[32,32],[64,64]]
    # dense_size_candidates = [[64],[128]]#[[64, 128],[128, 128],[128, 256]]
    # params={'conv_layer_sizes': conv_size_candidates,
    #                                 'dense_layer_sizes': dense_size_candidates,
    #                                  # epochs is avail for tuning even when not
    #                                  # an argument to model building function
    #                                  'epochs': [max_epochs],
    #                                  'kernel_size': [3,5],
    #                                  'pool_size': [2],
    #                                  'dropout': [0.25,0.5]}

    # best parameters, grid search learning rate
    conv_size_candidates = [[32,64]]
    dense_size_candidates = [[64]]#[[64, 128],[128, 128],[128, 256]]
    params={'conv_layer_sizes': conv_size_candidates,
                                    'dense_layer_sizes': dense_size_candidates,
                                     # epochs is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [max_epochs],
                                     'kernel_size': [5],
                                     'pool_size': [2],
                                     'dropout': [0.5],
                                     'lr': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}

   
    validator = GridSearchCV(my_classifier,
                         param_grid=params,
                         scoring=scoring,
                         n_jobs=1,
                         verbose=10, refit='f1_micro',return_train_score=True)
    validator.fit(x_train, y_train)
    
    print(validator.cv_results_)


    results = pd.DataFrame.from_dict(validator.cv_results_)


    export_csv = results.to_csv (saveDir + "/gridsearchResults.csv", index = None, header=True) #Don't forget to add '.csv' at the end of the path



   

    print('The parameters of the best model are: ')
    print(validator.best_params_)

    # validator.best_estimator_ returns sklearn-wrapped version of best model.
    # validator.best_estimator_.model returns the (unwrapped) keras model
    best_model = validator.best_estimator_
   
    # metric_names = best_model.metrics_names
    # metric_values = best_model.evaluate(x_train, y_train)
    # for metric, value in zip(metric_names, metric_values):
    #     print(metric, ': ', value)

    # serialize model to JSON
    print('Saving model to:', saveDir + '/model.json')
    model_json = best_model.to_json()
    with open(saveDir + "/model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    best_model.save_weights(saveDir + "/final_weights.hdf5")

   
    print("Keras Optimization Finished with in epochs, with", metric_values , "validation accuracy.")




def train_plotLearningRates(  data,
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
    
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Conv2D, Flatten,Dropout,MaxPooling2D,LeakyReLU,Activation,LocallyConnected2D
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


 
    #Network graph params
    filter_size_conv1 = 3 
    num_filters_conv1 = 32
    filter_size_conv2 = 3
    num_filters_conv2 = 32
    filter_size_conv3 = 3
    num_filters_conv3 = 64
    fc_layer_size = 128

    def make_model(conv_layer_sizes, dense_layer_sizes, kernel_size, pool_size,dropout, lr):
        '''Creates model comprised of 2 convolutional layers followed by dense layers
        dense_layer_sizes: List of layer sizes.
            This list has one number for each layer
        filters: Number of convolutional filters in each convolutional layer
        kernel_size: Convolutional kernel size
        pool_size: Size of pooling area for max pooling
        '''

        model = Sequential()
        # define input to the encoder and decoder model:
        inputs  = Input(shape=(img_size, img_size, num_channels))
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

        opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',#keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), #keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
       

        return model 

    # Load all the data in array
    x_train, y_train = data.train.nextBatch(data.train.num_examples, img_size)
    x_train = x_train[:,:,:,channels]

    x_valid, y_valid = data.test.nextBatch(data.test.num_examples, img_size)
    x_valid = x_valid[:,:,:,channels]


    # define safe directory
    outputDir = os.path.dirname(outputDir) ## directory of output model
    saveDir = os.path.join(outputDir,outputModel)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)


    possibleLearningRates = [0.0001, 0.001, 0.002, 0.004, 0.006, 0.007]

    import pandas as pd
    lossData = None#np.array([])
    valLossData = None#np.array([])
    accData = None#np.array([])
    valAccData = None#np.array([])


    for lr in possibleLearningRates:
        from numpy.random import seed
        seed(1)
        from tensorflow import set_random_seed
        set_random_seed(2)

        model = make_model(conv_layer_sizes=[32,64], dense_layer_sizes=[64], kernel_size=5, pool_size=2,dropout=0.5, lr=lr)


        history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=max_epochs,
                verbose=1,
                validation_data=(x_valid, y_valid),
                shuffle=True)


        history_dict = history.history

     

        loss = np.insert(np.asarray(history_dict['loss']), 0, lr, axis=0)
        loss = np.expand_dims(loss, axis=0)


        valloss = np.insert(np.asarray(history_dict['val_loss']), 0, lr, axis=0)
        valloss = np.expand_dims(valloss, axis=0)
        acc = np.insert(np.asarray(history_dict['accuracy']), 0, lr, axis=0)
        acc = np.expand_dims(acc, axis=0)
        valacc = np.insert(np.asarray(history_dict['val_accuracy']), 0, lr, axis=0)
        valacc = np.expand_dims(valacc, axis=0)


        if lossData is None:
            lossData = loss
        else:
            lossData = np.append(lossData, loss, axis=0)
        if valLossData is None:
            valLossData = valloss
        else:
            valLossData = np.append(valLossData, valloss, axis=0)
        if accData is None:
            accData = acc
        else:
            accData = np.append(accData, acc, axis=0)
        if valAccData is None:
            valAccData = valacc
        else:
            valAccData = np.append(valAccData, valacc, axis=0)

        np.savetxt(saveDir + "/lossData.csv", lossData, delimiter=",")
        np.savetxt(saveDir + "/valLossData.csv", valLossData, delimiter=",")
        np.savetxt(saveDir + "/accData.csv", accData, delimiter=",")
        np.savetxt(saveDir + "/valAccData.csv", valAccData, delimiter=",")

        

    # print("Keras Optimization Finished with in ", max_epochs, " epochs, with", valid_acc , "validation accuracy.")