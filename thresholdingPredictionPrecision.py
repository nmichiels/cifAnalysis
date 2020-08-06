import numpy as np
from cnn.prediction_keras import getConfusionMatrix, showConfusionMatrix, printAccuracyMetrics,saveConfusionMatrixFigure
classes = ['Functional', 'Exhausted']
# y_pred = np.load('y_pred.npy')
# y_true = np.load('y_true.npy')


classes = ["CD8+","CD4+","NK"]
channels =  [0,1,2,3,4,5]

# generate model name based on selected channels and classes
datasetDir = 'example/models/'
model = 'CD8-CD4-NK_'
for c in channels:
    model = model + 'C%d'%c
model = model + '_valid20'



y_pred = np.load(datasetDir + model + '/y_pred.npy')
y_true = np.load(datasetDir + model + '/y_true.npy')





def plotPrecisionAndRecall(y_pred, y_true, classes, outputFile = None):
    from cnn.prediction_keras import getAccuracyMetricsPerClass

    classToPlot = 1
    x = np.arange(0.50, 1.0, 0.01)
    precision = []
    recall = []
    f1 = []
    for threshold in x:
        pred_label = np.argmax(y_pred, axis=1)
        target_label = y_true.argmax(axis=1)

        newLabels = []
        newPred = []
        for i in range(y_pred.shape[0]):
            if y_pred[i, pred_label[i]] > threshold:
                newPred.append([y_pred[i,0],y_pred[i,1],0.0]) # keep predicted label
                newLabels.append([y_true[i,0],y_true[i,1], 0.0])
            else:
                newPred.append([0.0,0.0,1.0]) # set predicted label as unknown class
                newLabels.append([y_true[i,0],y_true[i,1], 0.0])

        newClasses = [classes[0],classes[1], 'Unknown']


        newPred = np.asarray(newPred)
        newLabels = np.asarray(newLabels)


        prec, rec, f1_score = getAccuracyMetricsPerClass(newLabels.argmax(axis=1), newPred.argmax(axis=1), newClasses)
        precision.append(prec[classToPlot])
        recall.append(rec[classToPlot])
        f1.append(f1_score[classToPlot])

    print(precision)
    import matplotlib as plt
    import matplotlib.pyplot as plt
    plt.plot(x, precision, label='precision')
    plt.plot(x, recall, label='recall')
    plt.plot(x, f1, label='f1 score')
    
    idx = np.argwhere(np.diff(np.sign(np.asarray(precision) - np.asarray(recall)))).flatten()

    if (len(idx) > 0):
        idx = idx[0]+1
        print(idx)
        plt.plot(x[idx], precision[idx], 'ro', label = 'precision ~ recall')
    
    idxMax = np.argmax(f1)
    
    plt.plot(x[idxMax], f1[idxMax], 'go', label = 'max f1 score')
    plt.grid(True)
    plt.xlabel('Threshold in %')
    plt.ylabel('%')
    plt.title('Change in precision and recall for different thresholds')
    plt.legend(loc='lower left')

    if (outputFile is not None):
        plt.savefig(outputFile, bbox_inches='tight')
    plt.show()



def plotNumberOfClassifiedCells(y_pred, y_true, classes, outputFile = None):
    from cnn.prediction_keras import getAccuracyMetricsPerClass

    classToPlot = 1
    x = np.arange(0.50, 1.0, 0.01)
    precision = []
    recall = []
    f1 = []

    truePos = []
    falsePos = []
    falseNeg = []
    for threshold in x:
        pred_label = np.argmax(y_pred, axis=1)
        target_label = y_true.argmax(axis=1)

        newLabels = []
        newPred = []
        for i in range(y_pred.shape[0]):
            if y_pred[i, pred_label[i]] > threshold:
                newPred.append([y_pred[i,0],y_pred[i,1],0.0]) # keep predicted label
                newLabels.append([y_true[i,0],y_true[i,1], 0.0])
            else:
                newPred.append([0.0,0.0,1.0]) # set predicted label as unknown class
                newLabels.append([y_true[i,0],y_true[i,1], 0.0])

        newClasses = [classes[0],classes[1], 'Unknown']


        newPred = np.asarray(newPred)
        newLabels = np.asarray(newLabels)

        numCorrectClassified = 0
        numWronglyClassified = 0
        numNotClassified = 0
        for i in range(newPred.shape[0]):
            label = newLabels[i,:].argmax()
            prediction = newPred[i,:].argmax()
            if label == classToPlot:
                if label == prediction:
                    numCorrectClassified = numCorrectClassified + 1
                else:
                    numNotClassified = numNotClassified + 1
            elif prediction == classToPlot:
                numWronglyClassified = numWronglyClassified + 1
            
        truePos.append(numCorrectClassified)
        falsePos.append(numWronglyClassified)
        falseNeg.append(numNotClassified)






        prec, rec, f1_score = getAccuracyMetricsPerClass(newLabels.argmax(axis=1), newPred.argmax(axis=1), newClasses)
        precision.append(prec[classToPlot])
        recall.append(rec[classToPlot])
        f1.append(f1_score[classToPlot])

    print(precision)
    import matplotlib as plt
    import matplotlib.pyplot as plt
    plt.plot(x, truePos, label='Correctly classified (true pos)', c='#2ca02c')
    plt.plot(x, falsePos, label='Wrongly classified as exhausted (false pos)', c='#d62728')
    plt.plot(x, falseNeg, label='Not classified as exhausted (false neg)', c='#ff7f0e')

    idx = np.argwhere(np.diff(np.sign(np.asarray(precision) - np.asarray(recall)))).flatten()
    if (len(idx) > 0):
        idx = idx[0]+1
        print(idx)
        plt.plot([x[idx],x[idx],x[idx]], [truePos[idx],falsePos[idx],falseNeg[idx]], 'o', c='#2ca02c',label='precision ~ recall')
        plt.axvline(x=x[idx],  c='#2ca02c')


    idxMax = np.argmax(f1)
    
    plt.plot([x[idxMax],x[idxMax],x[idxMax]], [truePos[idxMax],falsePos[idxMax],falseNeg[idxMax]], 'o', c='#7f7f7f', label = 'max f1 score')
    plt.axvline(x=x[idxMax], c='#7f7f7f')
    plt.yscale('log')

    

    plt.grid(True)
    plt.xlabel('Threshold in %')
    plt.ylabel('Number of cells (log scale)')
    plt.title('Change in Classification of Exhausted T-Cells for Different Thresholds')
    plt.legend(loc='lower left')

    if (outputFile is not None):
        plt.savefig(outputFile, bbox_inches='tight')

    plt.show()
    


    # plt.plot(x, precision, label='precision')
    # plt.plot(x, recall, label='recall')
    # plt.plot(x, f1, label='f1 score')
    
    # idx = np.argwhere(np.diff(np.sign(np.asarray(precision) - np.asarray(recall)))).flatten()[1]
    
    # idxMax = np.argmax(f1)
    # print(idx)
    # plt.plot(x[idx], precision[idx], 'ro', label = 'precision ~ recall')
    # plt.plot(x[idxMax], f1[idxMax], 'go', label = 'max f1 score')
    # plt.grid(True)
    # plt.xlabel('Threshold in %')
    # plt.ylabel('%')
    # plt.title('Change in precision and accuracy for different thresholds')
    # plt.legend(loc='down left')
    # plt.show()


plotPrecisionAndRecall(y_pred, y_true, classes, datasetDir + model + '/thresholdingPrecision.png')
plotNumberOfClassifiedCells(y_pred, y_true, classes, datasetDir + model + '/numberOfClassifiedCellsPerThreshold.png')
# import sys
# sys.exit(0)



thresholds = [0.50, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
# for threshold in np.arange(0.80, 0.99, 0.01):
    # increasing precision by thresholding?

for threshold in thresholds:
    # threshold = 0.69
    print("threshold:", threshold)
    pred_label = np.argmax(y_pred, axis=1)
    target_label = y_true.argmax(axis=1)

    newLabels = []
    newPred = []
    for i in range(y_pred.shape[0]):
        if y_pred[i, pred_label[i]] > threshold:
            newPred.append([y_pred[i,0],y_pred[i,1],0.0]) # keep predicted label
            newLabels.append([y_true[i,0],y_true[i,1], 0.0])
        else:
            newPred.append([0.0,0.0,1.0]) # set predicted label as unknown class
            newLabels.append([y_true[i,0],y_true[i,1], 0.0])

    newClasses = [classes[0],classes[1], 'Unknown']


    newPred = np.asarray(newPred)
    newLabels = np.asarray(newLabels)

  
    # Creating the Confusion Matrix
    cm = getConfusionMatrix(newLabels, newPred, np.arange(len(newClasses)))
    printAccuracyMetrics(newLabels.argmax(axis=1), newPred.argmax(axis=1), newClasses)
    saveConfusionMatrixFigure(cm, newClasses, datasetDir + model + '/cm_threshold_%d.png'%(int(100.0*threshold)))
    #showConfusionMatrix(cm, newClasses)








