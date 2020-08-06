import numpy as np
import matplotlib.pyplot as plt



def showConfusionMatrix(confusion, classes):
    
    import seaborn as sn
    import pandas as pd

    df_cm = pd.DataFrame(confusion, index = ["Actual " + str(x) for x in classes],
                  columns =  ["Predicted " + str(x) for x in classes])


    fig = plt.figure()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.title('Confusion Matrix')
    sn.heatmap(df_cm, annot=True, fmt='g',cmap='Blues')# font size


def showHistogram(predictionVals):
    H, bins = np.histogram(predictionVals)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig = plt.figure()
    fig.canvas.set_window_title('Output Class Histogram')
    plt.title('Output Class Histogram')
    plt.bar(center, H, align='center', width=width)
    plt.show()
