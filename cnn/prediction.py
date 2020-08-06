from dataset.dataset import Dataset
import numpy as np
import tensorflow as tf
from visualizations.visualizations import showConfusionMatrix, showHistogram
from visualizations.cvVisualizations import _rescale



def loadGraph(inputDir, inputModel):
    print("loading ", inputDir+inputModel)
    ## Let us restore the saved model 
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph(inputDir + inputModel)
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint(inputDir))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    return sess, graph

class DataSets(object):
        pass

def getConfusionMatrix(sess, y_true_cls, y_pred_cls, num_classes):
    confusion = tf.confusion_matrix(labels=y_true_cls, predictions=y_pred_cls, num_classes=num_classes)
    return sess.run(confusion)

def getAccuracyMetrics(sess, y_true_cls, y_pred_cls):
    recallTf = tf.metrics.recall(labels=y_true_cls, predictions=y_pred_cls)
    precisionTf = tf.metrics.precision(labels=y_true_cls, predictions=y_pred_cls)

    init_l = tf.local_variables_initializer()
    sess.run(init_l)
    recall = sess.run(recallTf)
    precision = sess.run(precisionTf)


    f1_score = 2*(recall[0] * precision[0]) / (recall[0] + precision[0])
    return precision, recall, f1_score

def getAccuracyMetricsPerClass(y_true_cls, y_pred_cls):
    from sklearn.metrics import precision_recall_fscore_support as score
    print(y_pred_cls.shape)
    precision, recall, f1_score, support = score(y_true_cls, y_pred_cls)
    return precision, recall, f1_score

def printAccuracyMetrics(y_true_cls, y_pred_cls, classes):
    precision, recall, f1_score = getAccuracyMetricsPerClass(y_true_cls, y_pred_cls)
    print("----------------------------------------------------")
    for idx, cls in enumerate(classes):
        print(cls)
        print("\tPrecision:\t", precision[idx])
        print("\tRecall:\t\t", recall[idx])
        print("\tF1:\t\t", f1_score[idx])
        print("----------------------------------------------------")

def predict( sess, graph, data, classes, channels, img_size = 35, batch_size = 256):
    
    numChannels = len(channels)
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_pred_cls = tf.argmax(y_pred, axis=1)

   
    # for op in tf.get_default_graph().as_graph_def().node:
    #     print(str(op.name))


    numError = 0
    total = 0
    total_batch = int(data.test.num_examples/batch_size)+1

    predictionVals = np.empty((0,data.test.num_classes), float)
    pred_cls_total = np.empty((0), float)
    true_cls_total = np.empty((0), float)
    # Loop over all batches
    faults = []
    for i in range(data.test.num_classes):
        faults.append(np.empty(( 0, img_size, img_size, numChannels)))

    np.empty((data.test.num_classes, 0, img_size, img_size, numChannels))
    for i in range(total_batch):
        x_batch_full, y_true_batch = data.test.next_batch(batch_size)
        x_batch = x_batch_full[:,:,:,channels]
        # x_batch = np.rot90(x_batch, 2, axes=(1,2))

        feed_dict_testing = {x: x_batch,
                        y_true: y_true_batch}
     
        prediction = sess.run(y_pred, feed_dict=feed_dict_testing)
        # print("iteration ",i)
        pred_cls = np.argmax(prediction, axis=1)
        true_cls = np.argmax(y_true_batch, axis=1)

        pred_cls_total = np.append(pred_cls_total, pred_cls, axis=0)
        true_cls_total = np.append(true_cls_total, true_cls, axis=0)
        predictionVals = np.append(predictionVals, prediction, axis=0)


        nonzeroind = np.nonzero((pred_cls-true_cls)<0)[0]
        faults[0] = np.concatenate((faults[0], x_batch[nonzeroind]), axis=0)

        nonzeroind = np.nonzero((pred_cls-true_cls)>0)[0]
        faults[1] = np.concatenate((faults[1], x_batch[nonzeroind]), axis=0)

        # print(true_cls)
        
        numError = numError + np.count_nonzero(pred_cls - true_cls, axis=0)
        total = total + batch_size
        # print(y_true_batch, " --> ", prediction)
        # summary_writer.add_summary(summary, epoch * total_batch + i)


    for i in range(data.test.num_classes):
        print("Number of wrong", classes[i], ":",faults[i].shape[0])
        np.save("./tmp/wrong_%s.npy" % classes[i] , faults[i])

    print("Accuracy: ", (total-numError) / total)
  
    confusion = getConfusionMatrix(sess, true_cls_total, pred_cls_total, data.test.num_classes)
    printAccuracyMetrics(true_cls_total, pred_cls_total, classes)

    showConfusionMatrix(confusion, classes)
    showHistogram(predictionVals)




def saliency( sess, graph, data, numchannels, img_size = 35, batch_size = 256):
    import saliency
    from matplotlib import pylab as P

    


    def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
        import matplotlib.pyplot as plt
        """
        Convert a 1-channel matrix of intensities to an RGB image employing a colormap.
        This function requires matplotlib. See `matplotlib colormaps
        <http://matplotlib.org/examples/color/colormaps_reference.html>`_ for a
        list of available colormap.
        Args:
            intensity (np.ndarray): array of intensities such as saliency.
            cmap (str): name of the colormap to use.
            normalize (bool): if True, will normalize the intensity so that it has
                minimum 0 and maximum 1.
        Returns:
            np.ndarray: an RGB float32 image in range [0, 255], a colored heatmap.
        """
        assert intensity.ndim == 2, intensity.shape
        intensity = intensity.astype("float")

        if normalize:
            intensity -= intensity.min()
            intensity /= intensity.max()

        cmap = plt.get_cmap(cmap)
        intensity = cmap(intensity)[..., :3]
        return intensity.astype('float32')


    def ShowGrayscaleImage(im, title='', ax=None):
        if ax is None:
            P.figure()
        P.axis('off')
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
        P.title(title)


    def WaitOnImagesToClose(titles):
        import cv2
        while(1):
            k = cv2.waitKey(200)
            if k == 27:
                break

            closeWindow = 1
            for title in titles:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:        
                    closeWindow = 0      
            if closeWindow:
                break
        cv2.destroyAllWindows()

    def ShowOverlayCV(im, overlay, title="", pos=[0,0]):
        import cv2
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(title, pos[0],pos[1]);

        backgroundImage = _rescale(im)
        backgroundImage = backgroundImage / np.amax(backgroundImage)
        # overlay = _rescale(overlay)
        # overlay = overlay / np.amax(overlay) - 0.5
        
        # overlay[overlay < 0] = 0
        # overlay = (backgroundImage*0.5 + overlay*0.5)
        # # print(overlay)
        images = [backgroundImage, backgroundImage, backgroundImage] 
        # images[2] = )
         
        # print("overlay", overlay.dtype)
        backgroundImage = np.float32(cv2.merge(images))
        combined =backgroundImage + overlay


        # print("backgroundImage", backgroundImage.dtype)
        # combined = cv2.addWeighted( backgroundImage, 0.7, overlay, 0.3, 0)

        cv2.imshow(title,combined)

    def ShowImageCV(im, title="", pos=[0,0]):
        import cv2
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(title, pos[0],pos[1]);
        cv2.imshow(title,im)

        

        
    def ShowGrayscaleImageCV(im, title="", pos=[0,0]):
        import cv2
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(title, pos[0],pos[1]);
        im = im - np.min(im)
        minRange = np.min(im)#-0.01*np.min(im)
        maxRange = np.max(im)#+0.01*np.max(im)
        multFactor = 1000000

        print("min:", minRange*multFactor)
        print("max:", maxRange*multFactor)


        def rescaleImage(im, minRange, maxRange):
            return (im - minRange) / (maxRange - minRange)

        def changeRange(x):
            maxRange=cv2.getTrackbarPos("Max", title) / float(multFactor)
            minRange=cv2.getTrackbarPos("Min", title) / float(multFactor)
            imgScaled = rescaleImage(im, minRange, maxRange)
            cv2.imshow(title,imgScaled)



        cv2.createTrackbar("Max", title,int(maxRange*multFactor),int(maxRange*multFactor),changeRange)
        cv2.createTrackbar("Min", title,int(minRange*multFactor),int(maxRange*multFactor),changeRange)
        cv2.imshow(title,im)
        changeRange(0)
        
    # y_pred = graph.get_tensor_by_name("y_pred:0")
    logits = graph.get_tensor_by_name("y_pred:0")
    images= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_pred_cls = tf.argmax(logits, axis=1)
        
    # images = tf.placeholder(tf.float32, shape=(None, img_size,img_size, numchannels))
    neuron_selector = tf.placeholder(tf.int32)
    y = logits[0][neuron_selector]
    # gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)
    guided_backprop_saliency =saliency.GuidedBackprop(graph, sess, y, images)
    for i in range(100):
        # Make a prediction. 
        x_batch, y_true_batch = data.test.next_batch(1)
      #  x_batch = np.rot90(x_batch, 2, axes=(1,2))
        prediction_class = sess.run(y_pred_cls, feed_dict = {images: x_batch})[0]

        # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
 
        # Compute the vanilla mask and the smoothed mask.
        # vanilla_mask_3d = gradient_saliency.GetMask(x_batch[0,:,:,:], feed_dict = {neuron_selector: prediction_class})
        smoothgrad_guided_backprop = guided_backprop_saliency.GetMask(x_batch[0,:,:,:], feed_dict = {neuron_selector: prediction_class})
        
        # print("smoothgrad_guided_backprop", smoothgrad_guided_backprop.shape)
        
        # cv2.imwrite("abs-saliency.jpg", abs_saliency)

        # Call the visualization methods to convert the 3D tensors to 2D grayscale.
        # vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d, 95)
        smoothgrad_guided_backprop_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_guided_backprop)

        # Set up matplot lib figures.
        # ROWS = 1
        # COLS = 2
        # UPSCALE_FACTOR   = 10
        # P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

        # Render the saliency masks.
        for i in range(data.test.num_channels):
            # ShowGrayscaleImageCV(x_batch[0,:,:,i], title='Input Channel C' + str(i), pos=[50+i*450,10])
            ShowImageCV(_rescale(x_batch[0,:,:,i]), title='Input C' + str(i), pos=[50+i*450,10])
            # print("vanilla_mask_3d", vanilla_mask_3d.shape)
            
            abs_saliency = np.abs(smoothgrad_guided_backprop[:,:,i])
            # print(abs_saliency.shape)
            pos_saliency = np.maximum(0, smoothgrad_guided_backprop[:,:,i])
            neg_saliency = np.maximum(0, -smoothgrad_guided_backprop[:,:,i])

            pos_saliency -= pos_saliency.min()
            pos_saliency /= pos_saliency.max()
            # cv2.imwrite('pos.jpg', pos_saliency * 255)

            neg_saliency -= neg_saliency.min()
            neg_saliency /= neg_saliency.max()
            # cv2.imwrite('neg.jpg', neg_saliency * 255)

            abs_saliency = intensity_to_rgb(abs_saliency, normalize=True)[:, :, ::-1]  # bgr
            # print("abs_saliency", abs_saliency)
    

            ShowImageCV(abs_saliency, title='Saliency C' + str(i), pos=[50+i*450,360])

            ShowOverlayCV(x_batch[0,:,:,i], abs_saliency, "Overlay C" + str(i), pos=[50+i*450,710])
        WaitOnImagesToClose(['Input C0']), 
            
        
        # ShowGrayscaleImage(x_batch[0,:,:,0], title='Input Image', ax=P.subplot(ROWS, COLS, 1))
        # ShowGrayscaleImage(vanilla_mask_grayscale, title='SmoothGrad', ax=P.subplot(ROWS, COLS, 2))
        # P.show()


def saliency2( sess, graph, data, batch_size = 256):
    
    from saliency.guided_backprop import GuidedBackprop
    from  saliency import visualization
    
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_pred_cls = tf.argmax(y_pred, axis=1)
    
    # Compute guided backprop.
    # NOTE: This creates another graph that gets cached, try to avoid creating many
    # of these.
    guided_backprop_saliency = GuidedBackprop(graph, sess, y_pred, x)

    for i in range(100):
        # Make a prediction. 
        x_batch, y_true_batch = data.test.next_batch(1)
      #  x_batch = np.rot90(x_batch, 2, axes=(1,2))
        prediction_class = sess.run(y_pred_cls, feed_dict = {x: x_batch})[0]

        print(prediction_class, y_true_batch)
        # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
        
        

        smoothgrad_guided_backprop = guided_backprop_saliency.GetMask(image)

        # Compute a 2D tensor for visualization.
        grayscale_visualization = visualization.VisualizeImageGrayscale( smoothgrad_guided_backprop)


        print(x_batch.shape)

def visualizeActivations( sess, graph, data, img_size, batch_size = 256):
    import math
    import matplotlib.pyplot as plt

    num_channels = data.test.num_channels

    y_pred = graph.get_tensor_by_name("y_pred:0")
    x= graph.get_tensor_by_name("x:0") 
    # y_true = graph.get_tensor_by_name("y_true:0") 
    y_pred_cls = tf.argmax(y_pred, axis=1)
    hidden_layer_1 = graph.get_tensor_by_name("hidden_layer_1:0") 
    hidden_layer_2 = graph.get_tensor_by_name("hidden_layer_2:0") 
    hidden_layer_3 = graph.get_tensor_by_name("hidden_layer_3:0") 

    def plotNNFilter(units, title):
        filters = units.shape[3]
        fig = plt.figure(figsize=(img_size,img_size))
        fig.canvas.set_window_title(title)
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

    def showActivations(layer,stimuli, img_size, num_channels, title):
        units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,img_size,img_size,num_channels],order='F')})
        plotNNFilter(units, title)


    
    for i in range(100):
        # Make a prediction. 
        x_batch, y_true_batch = data.test.next_batch(1)
      #  x_batch = np.rot90(x_batch, 2, axes=(1,2))
        prediction_class = sess.run(y_pred_cls, feed_dict = {x: x_batch})[0]

        print(prediction_class, y_true_batch)
        # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.


        print(x_batch.shape)


        imageToUse = x_batch[0,:,:,:]
        plt.figure()
        # show the first channel as reference
        plt.imshow(np.reshape(x_batch[0,:,:,0],[img_size,img_size]), interpolation="nearest", cmap="gray")
        
  
        showActivations(hidden_layer_1,imageToUse, img_size, num_channels, "Hidden Layer 1")
        showActivations(hidden_layer_2,imageToUse, img_size, num_channels, "Hidden Layer 2")
        showActivations(hidden_layer_3,imageToUse, img_size, num_channels, "Hidden Layer 3")
        plt.show()
