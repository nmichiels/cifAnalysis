## CIF Analysis
**CIF Analysis** is a framework for deep learning on cif datasets, captured with the Amnis Imagestreamer platform.

## Installation
Training and predicting are working with the latest Tensorflow version 2.3.0 and 2.4.3
Unfortunately, the saliency analysis of dependency `keras-vis` only works with Tensorflow <= 1.15.3 and Keras <= 2.3.0.
The requirements file in this repository contains the dependencies for a working saliency analysis. Feel free to update tensorflow to the latest version for more performant training.
Tested with Python 3.7


Make sure to pull the [CifDataset](https://github.com/nmichiels/cifDataset) submodule `git pull --recurse-submodules` and follow the install instructions of the readme of the submodule.

Install all the requirements:
```
pip install -r requirements.txt
```


For saliency analysis and to avoid dependency erros, install tensorflow, keras-vis and keras in the following order:
```
pip install -U -I git+https://github.com/raghakot/keras-vis.git
pip install tensorflow-gpu==1.15.3
pip install keras==2.3.0
```


## Training a classifier on the example dataset
The example dataset can be found in `./example`.
First run `./convertCIF.sh` in the example folder to convert the cif files to hdf5 datasets and/or numpy arrays.

Training the dataset can be done by executing the folowing python script:
```
python .\train_classifier_example.py
```
This will load the example dataset and train a classifier. The trained model will be saved in `./example/models/`.
Predicting data using this model can be done by executing a different python script:
```
python .\predict_classifier_example.py
```
Visualizing saliency or activation layers can be done with the following script:
```
python .\visualize_example.py
```

PCA and tSNE analysis can be performed by:
```
python .\tsne_example.py
python .\pca_example.py
```


Alternative training and predictions are commented out in the respective script.


## Training different kind of models/
Training a convolutional autoencoder (not enough data in example dataset to make it work). For testing, you can enable the mnist dataset:
```
python .\train_ae_example.py
```

Training a siamese network classifier:
```
python .\train_siamese_example.py
```

Training a deepclustering approach. Sadly this model is not stable and does only work on mnist:
```
python .\train_deepclustering_example.py
```

## Visualizing datasets
These visualization require opencv: `pip install opencv-python`

Examples on how to visualize data using opencv are given in this scripts:
```
python showDataset.py
```
