import numpy as np
from cifDataset.cifStreamer.npDataset import NPDataset
from cifDataset.cifStreamer.npDataset_nolabels import NPDataset_nolabels
from cifDataset.cifStreamer.hdf5Dataset import HDF5Dataset
from cifDataset.cifStreamer.dataset import Dataset
from dataset.trainingDataset import TrainingDataset




class DataSets(object):
  pass

def loadFile(file):
  if file.endswith('.npy') or file.endswith('.np') or file.endswith('.numpy'):
    print("Numpy file")
    return NPDataset_nolabels(np.load(file))#[:,:,:,channels])
  elif file.endswith('.hdf5') or file.endswith('.h5') or file.endswith('.hdf') or file.endswith('.he5'):
    print("hdf5 file")
    return HDF5Dataset(file)
  elif file.endswith('.cif'):
    print("CIF file")
  elif file.endswith('.rif'):
    print("RIF file")

def loadCrossValidationDatasetIndexed(files, classes,channels, img_size, validation_size = 0.2, max_num_examples = None, rotateInvariant = False):
  np.random.seed(0)

  num_classes = len(classes)
  num_channels = len(channels)


  

  datasets = [loadFile(file) for idx, file in enumerate(files)]

  trainingDataset = TrainingDataset()
  trainingDataset.generate(datasets, classes)

  if max_num_examples:
      trainingDataset.setMaxNumberExamples(max_num_examples)

  if isinstance(validation_size, float):
        validation_size = int(validation_size * trainingDataset._num_examples)

  train = trainingDataset.getSubset(validation_size, trainingDataset._num_examples)
  valid = trainingDataset.getSubset(0, validation_size)

  data = DataSets()
  data.train = train
  data.test = valid

  data.img_size = img_size


  print("Complete preparing cross validation data.")
  print("Number of files in Training Dataset:\t\t{}".format(data.train.num_examples))
  print("Number of files in Validation Dataset:\t\t{}".format(data.test.num_examples))
  print("Number of Channels:\t\t\t\t{}".format(trainingDataset.num_channels))
  print("Number of Classes:\t\t\t\t{}".format(trainingDataset._num_classes))
  print("Image Size:\t\t\t\t\t{}".format(img_size))
  return data


def loadTestDatasetIndexed(files, classes,channels, img_size, validation_size = 0.2, max_num_examples = None, rotateInvariant = False):
  np.random.seed(0)

  num_classes = len(classes)
  num_channels = len(channels)
  datasets = [loadFile(file) for idx, file in enumerate(files)]

  trainingDataset = TrainingDataset()
  trainingDataset.generate(datasets, classes)
  
  if max_num_examples:
      trainingDataset.setMaxNumberExamples(max_num_examples)

  data = DataSets()
  data.test = trainingDataset

  data.img_size = img_size

  print("Complete preparing test dataset.")
  print("Number of files in Dataset:\t\t\t{}".format(trainingDataset._num_examples))
  print("Number of Channels:\t\t\t\t{}".format(trainingDataset.num_channels))
  print("Number of Classes:\t\t\t\t{}".format(trainingDataset._num_classes))
  print("Image Size:\t\t\t\t\t{}".format(img_size))


  return data




def loadData(files, classes,channels, img_size, rotateInvariant = False, random=True, maxImages = None):

  num_classes = len(classes)
  num_channels = len(channels)

  print("num classes", num_classes)
  print("num channels", num_channels)
  # Load datafiles per class
  data = None
  if maxImages:
    data = [np.load(file)[:maxImages,:,:,:] for idx, file in enumerate(files)]
  else:
    data = [np.load(file) for idx, file in enumerate(files)]
  #data = [np.load(file)[:,:,:,channels] for idx, file in enumerate(files)] #TODO
  # Print data info

  for idx, cls in enumerate(classes):
      if len(data) > idx:
        print("Number of",cls, ":", data[idx].shape)
      else:
        print("Number of",cls, ":", 0)
  

  # Concatenate data for all classes
  images = np.concatenate((data), axis=0)

  # Generate label data
  labels = np.empty(0)
  for idx, cls in enumerate(classes):
      labels = np.append(labels, np.full(data[idx].shape[0], idx), axis=0)

  from keras.utils import to_categorical
  labels = to_categorical(labels)
 
  # # Generate label data
  # labels = np.empty([len(images),len(classes)])
  # for idx_cls in range(len(classes)):
  #     labels_cls = np.array([])
  #     for idx in range(len(classes)):
  #         if (idx_cls == idx):
  #             labels_cls = np.concatenate((labels_cls, np.ones(data[idx].shape[0])), axis=0)
  #         else:
  #             labels_cls = np.concatenate((labels_cls, np.zeros(data[idx].shape[0])), axis=0)
  #     labels[:,idx_cls] = labels_cls

  # optionally make the datset rotation invariant by appending rotations of 90, 180 and 270 degrees
  if (rotateInvariant == True):
        #append dataset with rotations of 90 180 and 270 degrees
    allRotatedImages = np.append(images, np.rot90(images, 1, axes=(1,2)), axis=0)
    allRotatedImages = np.append(allRotatedImages, np.rot90(images, 2, axes=(1,2)), axis=0)
    allRotatedImages = np.append(allRotatedImages, np.rot90(images, 3, axes=(1,2)), axis=0)
    print("Making dataset Rotation Invariant:", images.shape[0], "to", allRotatedImages.shape[0], "rotated images.")
    images = allRotatedImages
    labels = np.concatenate((labels, labels, labels, labels), axis=0)
    


  if (random == True):
    p = np.random.permutation(images.shape[0])
    images = images[p]
    labels = labels[p]

  return images, labels



def loadDataset(files,
            classes,
            channels,
            img_size
):
  print("****Initializing Dataset****")
  np.random.seed(0)
  images, labels = loadData(files, classes,channels, img_size, False, False)
    

  # Creating datasets
  data = NPDataset(images, labels)
  data.img_size = img_size
  print("Complete reading input data.")
  print("Number of files in Dataset:\t\t{}".format(len(images)))
  return data



def loadTestDataset(files,
            classes,
            channels,
            img_size,
            rotateInvariant = False,
            random = True
):
  print("****Initializing Dataset****")
  np.random.seed(0)
  images, labels = loadData(files, classes,channels, img_size, rotateInvariant, random)
    

  # Creating datasets
  data = DataSets()
  data.test = NPDataset(images, labels)
  data.img_size = img_size

  print("Complete reading input data.")
  print("Number of files in Dataset:\t\t{}".format(len(images)))
  return data


def loadCrossValidationDataset(files, classes, channels, img_size, validation_size = 0.2, max_num_images = -1, files_masked = None, rotateInvariant = False, random = True):
  print("****Initializing Dataset****")
  np.random.seed(0)
  images, labels = loadData(files, classes,channels, img_size, rotateInvariant, random, maxImages=max_num_images)


   

  # Subdivide data in training and validation data
  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])


  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]



  print("Shape of training data:   ", train_images.shape)
  print("Shape of validation data: ", validation_images.shape)


  if files_masked is not None:
    np.random.seed(0)
    print("Loading masks")
    [images_masks,_]  = loadData(files_masked, classes,channels, img_size, rotateInvariant)
    validation_images_masks = images_masks[:validation_size]
    train_images_masks = images_masks[validation_size:]
    print("Shape of training mask data:   ", train_images_masks.shape)
    print("Shape of validation mask data: ", validation_images_masks.shape)

  
    # train_images = np.rot90(train_images, 2, axes=(1,2))
    # validation_images = np.rot90(validation_images, 2, axes=(1,2))
  # Creating datasets
  data = DataSets()
  data.train = NPDataset(train_images, train_labels)
  data.test = NPDataset(validation_images, validation_labels)
  if files_masked is not None:
    data.trainMasked = NPDataset(train_images_masks, train_labels)
    data.testMasked = NPDataset(validation_images_masks, validation_labels)
  data.img_size = img_size
  return data



def loadMNISTDataset(fraction):
  from keras.datasets import mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  img_size = x_train.shape[1]

  # extract subset of images
  dataset_size = int(fraction * x_train.shape[0])
  x_train = x_train[0:dataset_size,:,:]
  x_test = x_test[0:dataset_size,:,:]
  y_train = y_train[0:dataset_size]
  y_test = y_test[0:dataset_size]



  x_train = np.reshape(x_train, (len(x_train), img_size, img_size, 1)) 
  x_test = np.reshape(x_test, (len(x_test), img_size, img_size, 1)) 

  classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

  labels_train = np.zeros((y_train.shape[0], classes.shape[0]))
  for (index,_) in enumerate(y_train):    
    labels_train[index,y_train[index]] = 1
  
  labels_test = np.zeros((y_test.shape[0], classes.shape[0]))
  for (index,_) in enumerate(y_test):    
    labels_test[index,y_test[index]] = 1


  data = DataSets()
  data.train = NPDataset(x_train, labels_train)
  data.test = NPDataset(x_test, labels_test)
  data.img_size = img_size
  
  return data, classes, img_size 


  
   