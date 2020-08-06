from dataset.dataset import Dataset
import numpy as np

#from visualizations.visualizations import showHistogram



class DataSets(object):
        pass


def saveConfusionMatrixFigure(confusion, classes, outputFile):
    import matplotlib.pyplot as plt
    import seaborn as sn
    import pandas as pd

    # df_cm = pd.DataFrame(confusion, index = ["Actual " + str(x) for x in classes],
    #             columns =  ["Predicted " + str(x) for x in classes])
    df_cm = pd.DataFrame(confusion, index = ["" + str(x) for x in classes],
                columns =  ["" + str(x) for x in classes])

    fig = plt.figure()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.title('Confusion Matrix')
    sn.heatmap(df_cm, xticklabels=True, yticklabels=True, annot=True, fmt='g',cmap='Blues')# font size
    plt.savefig(outputFile, bbox_inches='tight')
    # plt.xticks(rotation=90)
    # plt.show()


def showConfusionMatrix(confusion, classes):
    import matplotlib.pyplot as plt
    import seaborn as sn
    import pandas as pd

    # df_cm = pd.DataFrame(confusion, index = ["Actual " + str(x) for x in classes],
    #             columns =  ["Predicted " + str(x) for x in classes])
    df_cm = pd.DataFrame(confusion, index = ["" + str(x) for x in classes],
                columns =  ["" + str(x) for x in classes])

    fig = plt.figure()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.title('Confusion Matrix')
    sn.heatmap(df_cm, xticklabels=True, yticklabels=True, annot=True, fmt='g',cmap='Blues')# font size
    
    # plt.xticks(rotation=90)
    plt.show()

def getAccuracyMetricsPerClass(y_true_cls, y_pred_cls, labels = None):
    from sklearn.metrics import precision_recall_fscore_support as score
    #print(y_pred_cls.shape)
    precision, recall, f1_score, support = score(y_true_cls, y_pred_cls, labels = np.arange(len(labels)))
    return precision, recall, f1_score

def printAccuracyMetrics(y_true_cls, y_pred_cls, classes):
    precision, recall, f1_score = getAccuracyMetricsPerClass(y_true_cls, y_pred_cls, classes)
    print("----------------------------------------------------")
    for idx, cls in enumerate(classes):
        print(cls)
        print("\tPrecision:\t", precision[idx])
        print("\tRecall:\t\t", recall[idx])
        print("\tF1:\t\t", f1_score[idx])
        print("----------------------------------------------------")


def getConfusionMatrix(y_true, y_pred, classes = None):
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1), labels = classes)


# set validation==True if input dataset contains labels and you want to validate the results
# set validation==False if input is unknown and no validation is required, only prediction
# set returnFeatures to True is you want optionally to return the extracted features
def predict( data, classes, channels, inputDir, inputModel, validation=True, returnFeatures=False, weights_file = 'final_weights.h5', img_size = 35, batch_size = 256, useGenerator = True):
    import keras
    from .keras_utils import My_Generator, My_GeneratorIncremental

    import os
    import errno
    from keras.models import model_from_json, Model
    from colorama import Fore, Back, Style
    

    print(Fore.GREEN + "\nStart predicting...")
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



    # define safe directory
    inputDir = os.path.dirname(inputDir) ## directory of input model
    loadDir = os.path.join(inputDir,inputModel)
    if not os.path.exists(loadDir):
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), loadDir)



    # load json and create model
    json_file = open(loadDir + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(loadDir + "/" + weights_file)
    print("Loaded model from disk")


    model.compile(  loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

    if returnFeatures:
        # define model for extracting activation features
        model.summary()
        layer_name = 'activations'
        activation_layer_model = Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
    


   
    #useGenerator = False
    valid_loss = 0
    valid_acc = 0
    y_pred = None
    activation_features = None
    if useGenerator:
        my_validation_batch_generator = My_Generator(data.test, batch_size, img_size, channels)

        valid_loss, valid_acc =  model.evaluate_generator(generator=my_validation_batch_generator, 
                                                        steps=(data.test.num_examples // batch_size + 1), 
                                                        max_queue_size = 32,
                                                        use_multiprocessing=False,
                                                        workers = 1)
        

        # Predicting the Test set results
        y_pred = model.predict_generator(generator=my_validation_batch_generator, 
                                                        steps=(data.test.num_examples // batch_size + 1 ), 
                                                        max_queue_size = 32,
                                                        use_multiprocessing=False,
                                                        workers = 1)

        if returnFeatures:
            activation_features = activation_layer_model.predict_generator(generator=my_validation_batch_generator, 
                                                            steps=(data.test.num_examples // batch_size + 1 ), 
                                                            max_queue_size = 32,
                                                            use_multiprocessing=False,
                                                            workers = 1)

    else:
        # Load all the data in array
        x_valid, y_valid = data.test.nextBatch(data.test.num_examples, img_size)
        x_valid = x_valid[:,:,:,channels]

        
        # global standardization
        # stds = x_valid.std(axis=(0,1,2), dtype='float64')
        # means = x_valid.mean(axis=(0,1,2), dtype='float64')
        # for i, img in enumerate(x_valid):
        #     for c in range(x_valid.shape[-1]):
        #         # print(c)
        #         x_valid[i,:,:,c] = (img[:,:,c] - means[c]) / stds[c]


        # # Local Standardization
        # # calculate per-channel means and standard deviations
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

    
        
        

        # Predicting the Test set results
        y_pred = model.predict(x=x_valid, batch_size=batch_size)


        if returnFeatures:
            activation_features = activation_layer_model.predict(x=x_valid, batch_size=batch_size)

        increasing_precision = False
        if increasing_precision and validation:
            valid_loss, valid_acc =  model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)

            # increasing precision by thresholding?
            threshold = 0.80
            pred_label = np.argmax(y_pred, axis=1)
            target_label = data.test.labels.argmax(axis=1)
            
            newLabels = []
            newPred = []
            for i in range(y_pred.shape[0]):
                if y_pred[i, pred_label[i]] > threshold:
                    newPred.append([y_pred[i,0],y_pred[i,1],0.0]) # keep predicted label
                    newLabels.append([data.test.labels[i,0],data.test.labels[i,1], 0.0])
                else:
                    newPred.append([0.0,0.0,1.0]) # set predicted label as unknown class
                    newLabels.append([data.test.labels[i,0],data.test.labels[i,1], 0.0])

            newClasses = [classes[0],classes[1], 'Unknown']
    

            newPred = np.asarray(newPred)
            newLabels = np.asarray(newLabels)

            print('newLabels', newLabels.shape)
            print('newPred', newPred.shape)
            # Creating the Confusion Matrix
            cm = getConfusionMatrix(newLabels, newPred)
            print(cm)
            printAccuracyMetrics(newLabels.argmax(axis=1), newPred.argmax(axis=1), newClasses)
            showConfusionMatrix(cm, newClasses)
            



    if (validation):
        print("Accuracy: ", valid_acc)
    
        # Creating the Confusion Matrix
        cm = getConfusionMatrix(data.test.labels, y_pred)
        print(cm)
        printAccuracyMetrics(data.test.labels.argmax(axis=1), y_pred.argmax(axis=1), classes)
        showConfusionMatrix(cm, classes)

    return activation_features, y_pred, data.test.labels



def extractActivationFeatures( data, classes, channels, inputDir, inputModel, weights_file = 'final_weights.h5', img_size = 35, batch_size = 256, outputActivationFeaturesFile = None):
    import os
    import errno
    import keras
    from .keras_utils import My_Generator, My_GeneratorIncremental
    from keras.models import model_from_json
    from colorama import Fore, Back, Style
    

    print(Fore.GREEN + "\nStart predicting...")
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



    # define safe directory
    inputDir = os.path.dirname(inputDir) ## directory of input model
    loadDir = os.path.join(inputDir,inputModel)
    if not os.path.exists(loadDir):
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), loadDir)



    # load json and create model
    json_file = open(loadDir + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(loadDir + "/" + weights_file)
    print("Loaded model from disk")


    

    # define model for extracting activation features
    model.summary()
    layer_name = 'activations'
    activation_layer_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
   
    activation_layer_model.compile(  loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

   
    useGenerator = False

    activation_features = None
    if useGenerator:
        my_validation_batch_generator = My_Generator(data.test, batch_size, img_size, channels)

   
        activation_features = activation_layer_model.predict_generator(generator=my_validation_batch_generator, 
                                                        steps=(data.test.num_examples // batch_size + 1 ), 
                                                        max_queue_size = 32,
                                                        use_multiprocessing=False,
                                                        workers = 1)

    else:
        import pickle
        # Load all the data in array
        
        x_test, y_test = data.test.nextBatch(data.test.num_examples, img_size)
        x_test = x_test[:,:,:,channels]


        activation_features = activation_layer_model.predict(x=x_test, batch_size=batch_size)
        if outputActivationFeaturesFile is not None:
            output_features_path = './prediction_keras_features.pickle'
            output_labels_path = './prediction_keras_labels.pickle'
            y_test = np.array(classes)[np.argmax(data.test.labels, axis=1)]
            pickle.dump(activation_features, open(output_features_path, 'wb'))
            pickle.dump(y_test, open(output_labels_path, 'wb'))

        

    


    print(activation_features)
    print(activation_features.shape)
    return activation_features
    # showHistogram(predictionVals)

    
def visualizeConvolutions( inputDir, inputModel, classes, channels, channelNames, weights_file = 'final_weights.h5', img_size=35):

    import os
    import errno
    import keras
    from .keras_utils import My_Generator, My_GeneratorIncremental
    from keras.models import model_from_json
    from colorama import Fore, Back, Style
    from visualizations.cvVisualizations import _rescale
    import vis
    from vis.utils import utils
    from keras import activations
    from vis.visualization import get_num_filters
    from vis.visualization import visualize_activation
    import matplotlib.pyplot as plt
    import math

    # define safe directory
    inputDir = os.path.dirname(inputDir) ## directory of input model
    loadDir = os.path.join(inputDir,inputModel)
    if not os.path.exists(loadDir):
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), loadDir)



    # load json and create model
    json_file = open(loadDir + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(loadDir + "/" + weights_file)
    print("Loaded model from disk")


    # define model for extracting activation features
    model.summary()


    numChannels = len(channels)
    numClasses = len(classes)


    # The name of the layer we want to visualize
    # You can see this in the model definition.
    layer_name = 'conv2d_2'
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.arange(get_num_filters(model.layers[layer_idx]))
    print(len(filters))
    # Generate input image for each filter.
    n_rows = len(filters)
    n_columns = numChannels
    fig, axs = plt.subplots(n_rows,n_columns, figsize=(img_size, img_size))

    for i in range(len(filters)):
        idx = filters[i]
        img = visualize_activation(model, layer_idx, filter_indices=idx)
        
        for c in range(numChannels):
            axs[i, c].imshow(img[:,:,c], interpolation="nearest", cmap="gray")
            axs[i,c].get_xaxis().set_ticks([])
            axs[i,c].get_yaxis().set_ticks([])
      


    for c in range(numChannels):
        channelName = "Channel " + str(c) + ' (' + channelNames[c] + ')'
        plt.setp(axs[-1, c], xlabel=channelName)
    #plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)

        
    plt.show()




def visualizeActivations( inputDir, inputModel, classes, channels, channelNames, weights_file = 'final_weights.h5', img_size=35):

    import os
    import errno
    import keras
    from .keras_utils import My_Generator, My_GeneratorIncremental
    from keras.models import model_from_json
    from colorama import Fore, Back, Style
    from visualizations.cvVisualizations import _rescale
    import vis
    from vis.utils import utils
    from keras import activations
    from vis.visualization import get_num_filters
    from vis.visualization import visualize_activation
    from vis.input_modifiers import Jitter
    import matplotlib.pyplot as plt
    import math

    # define safe directory
    inputDir = os.path.dirname(inputDir) ## directory of input model
    loadDir = os.path.join(inputDir,inputModel)
    if not os.path.exists(loadDir):
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), loadDir)



    # load json and create model
    json_file = open(loadDir + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(loadDir + "/" + weights_file)
    print("Loaded model from disk")


    # define model for extracting activation features
    model.summary()



    # The name of the layer we want to visualize
    # You can see this in the model definition.
    layer_name = 'activation_1'
    layer_idx = -1 #utils.find_layer_idx(model, layer_name)

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    numChannels = len(channels)
    numClasses = len(classes)
    #fig = plt.figure(figsize=(img_size,img_size))
    
    
    n_columns = numChannels
    n_rows = numClasses
    fig, axs = plt.subplots(n_rows,n_columns, figsize=(img_size, img_size))
    fig.canvas.set_window_title("Activations per Output Category")

    for idx in range(numClasses):
        img = visualize_activation(model, layer_idx, filter_indices=idx)
        
        for c in range(numChannels):
            axs[idx, c].imshow(img[:,:,c], interpolation="nearest", cmap="gray")
            axs[idx,c].get_xaxis().set_ticks([])
            axs[idx,c].get_yaxis().set_ticks([])
      

    # set labels
    for idx in range(numClasses):
        plt.setp(axs[idx, 0], ylabel=classes[idx])

    for c in range(numChannels):
        channelName = "Channel " + str(c) + ' (' + channelNames[c] + ')'
        plt.setp(axs[-1, c], xlabel=channelName)
    #plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)

        
    plt.show()


   







def visualizeSaliency( data, classes, channels, inputDir, inputModel, weights_file = 'final_weights.h5', channelNames = [], img_size = 35, batch_size = 256):

    import os
    import errno
    import keras
    from .keras_utils import My_Generator, My_GeneratorIncremental
    from keras.models import model_from_json
    from colorama import Fore, Back, Style
    from visualizations.cvVisualizations import _rescale
    

    print(Fore.GREEN + "\nStart predicting...")
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



    # define safe directory
    inputDir = os.path.dirname(inputDir) ## directory of input model
    loadDir = os.path.join(inputDir,inputModel)
    if not os.path.exists(loadDir):
        raise NotADirectoryError(errno.ENOENT, os.strerror(errno.ENOENT), loadDir)



    # load json and create model
    json_file = open(loadDir + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(loadDir + "/" + weights_file)
    print("Loaded model from disk")


    

    # define model for extracting activation features
    model.summary()
    model_orig = model

    model_orig.compile(  loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])



    x_test, y_test = data.test.nextBatch(data.test.num_examples, img_size)

    # #global standardization
    # stds = x_test.std(axis=(0,1,2), dtype='float64')
    # means = x_test.mean(axis=(0,1,2), dtype='float64')
    # for i, img in enumerate(x_test):
    #     for c in range(x_test.shape[-1]):
    #         # print(c)
    #         x_test[i,:,:,c] = (img[:,:,c] - means[c]) / stds[c]


    x_test = x_test[:,:,:,channels]
    # y_test = np.argmax(data.test.labels, axis=1)
    # y_test_labels = np.array(classes)[np.argmax(data.test.labels, axis=1)]


    class_idx = 0 # for IFNy+
    
    indices = np.where(y_test[:,class_idx] == 1.)[0]

    # pick some random input from here.
    idx = indices[1]

    
    


    # https://github.com/raghakot/keras-vis/blob/master/examples/mnist/attention.ipynb
    import vis
    from vis.visualization import visualize_saliency
    from vis.utils import utils
    from keras import activations
    from matplotlib import pyplot as plt

    # Utility to search for layer index by name. 
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = -1#utils.find_layer_idx(model, 'preds')

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

   

  


    plt.ion() # interactive mode
    plt.show()

    classAxs = []
    classFig = []

    classIndices = [] #[np.where(y_test[:, class_idx] == 1.)[0] for classIdx in np.arange(num_classes)]
    for class_idx in range(num_classes):
        if numChannels == 1:
            fig, axs = plt.subplots(numChannels+1, 2)
        else:
            fig, axs = plt.subplots(numChannels, 2)
        fig.suptitle(classes[class_idx])
        axs[0][0].set_title('Input ' + ' (???)')
        axs[0][1].set_title('Saliency')

        for c in range(numChannels):
                axs[c][0].set_ylabel(channelNames[channels[c]] + ' [%d]' %channels[c], rotation='horizontal', labelpad=40)

                axs[c][0].tick_params(
                            axis='both',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            left=False,         # ticks along the left edge are off
                            labelbottom=False,
                            labelleft = False) # labels along the bottom edge are off

                axs[c][1].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    left=False,         # ticks along the left edge are off
                    labelbottom=False,
                    labelleft = False) # labels along the bottom edge are off

        classAxs.append(axs)
        classFig.append(fig)
        classIndices.append(np.where(y_test[:, class_idx] == 1.)[0])


    maxImages = 50



    # iterate over multiple images and draw for each class the input image and its saliency map
    for i in range(maxImages):   
        
  

        for class_idx in range(num_classes):    
            
            idx = classIndices[class_idx][i] # get i'th image out of class with class_idx
            inputImg = x_test[idx:idx+1,:,:,:]
            # get prediction of idx
            y_pred = model_orig.predict(x=x_test[idx:idx+1,:,:,:], batch_size=1)
            true_cls = y_test[idx:idx+1,:].argmax(axis=1)
            pred_cls_idx = y_pred.argmax(axis=1)[0]
            # pred_cls_idx = cls_y_preds[class_idx][i]
            
            axs = classAxs[class_idx]
            classFig[class_idx].suptitle("Input Class: " + classes[class_idx] + "      -->      Predicted Class: " + classes[pred_cls_idx])
            axs[0][0].set_title('Input ' + ' [' + str(idx) + ']')
       
            # grads = cls_grads[class_idx][i]
            grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
                                        seed_input=inputImg, backprop_modifier='guided', keepdims=True)

                        
            # Stretch the intensity scale to visible 8-bit images
            def _rescale(image):
                import skimage.io
                import scipy.stats
                vmin, vmax = scipy.stats.scoreatpercentile(image, (0.01, 99.95))

                return skimage.exposure.rescale_intensity(image, in_range=(vmin, vmax), out_range=np.uint8).astype(np.uint8)

          
            for c in range(numChannels):  
                axs[c,0].imshow(_rescale(inputImg[0,:,:, c]))
                axs[c,1].imshow(grads[:,:,c], vmin=0.0, vmax=1.0, cmap='jet')
        plt.draw()
        plt.pause(0.001)
        plt.waitforbuttonpress()
        # input("Press [enter] to continue.")

        
