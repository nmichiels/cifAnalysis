import numpy as np
from cifDataset.cifStreamer.dataset import Dataset


def to_categorical(y, num_classes=None, dtype='float32'):
  """Converts a class vector (integers) to binary class matrix.
  E.g. for use with categorical_crossentropy.
  # Arguments
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes.
      dtype: The data type expected by the input, as a string
          (`float32`, `float64`, `int32`...)
  # Returns
      A binary matrix representation of the input. The classes axis
      is placed last.
  # Example
  ```python
  # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
  > labels
  array([0, 2, 1, 2, 0])
  # `to_categorical` converts this into a matrix with as many
  # columns as there are classes. The number of rows
  # stays the same.
  > to_categorical(labels)
  array([[ 1.,  0.,  0.],
          [ 0.,  0.,  1.],
          [ 0.,  1.,  0.],
          [ 0.,  0.,  1.],
          [ 1.,  0.,  0.]], dtype=float32)
  ```
  """

  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
      num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


# initialize new dataset with only a specific subset of indices
# used for created cross validation datasets
# lowerbound,upperbound = [0,1]  For example [0.8,1.0] for 20% validation size

class TrainingDataset(Dataset):
  
  def __init__(self):
    Dataset.__init__(self)

  def generate(self, datasets, labels):

    assert(len(datasets) == len(labels))
    assert(all(dataset.num_channels == datasets[0].num_channels for dataset in datasets))


    self._numDatasets = len(datasets)
    self._datasets = datasets
    self._num_channels = datasets[0].num_channels
    self._num_classes = len(labels)

    # prepare matrix of labels
    labels_dataset = [ np.full(dataset.num_examples,i) for i, dataset in enumerate(datasets)]
    self._labels = np.concatenate(labels_dataset, axis=0)
    self._labels = to_categorical(self._labels , self._num_classes)
 
    # prepare array of indices
    indices_dataset = [ np.column_stack((np.full(dataset.num_examples,i), np.arange(dataset.num_examples))) for i, dataset in enumerate(datasets)]
    self._indices = np.concatenate(indices_dataset, axis=0)

    self._num_examples = self._indices.shape[0]
    
    self.permutate()

  def getSubset(self, startIdx, endIdx):
    import copy
    subset = copy.copy(self)

    subset._indices = self._indices[startIdx:endIdx]
    subset._labels = self._labels[startIdx:endIdx]
    subset._num_examples = subset._indices.shape[0]
    return subset


  def setMaxNumberExamples(self, max_num_examples):
    self._num_examples = max_num_examples
    self._indices = self._indices[:max_num_examples]
    self._labels = self._labels[:max_num_examples]
 

  @property
  def labels(self):
    return self._labels

    
  def permutate(self):
    p = np.random.permutation(self._indices.shape[0])
    self._indices = self._indices[p]
    self._labels = self._labels[p]

  def get_batch(self, idx, batch_size, image_size):

    start = idx*batch_size
    end = (idx+1)*batch_size
    if end > self._num_examples:
        end = self._num_examples

    count = end-start

    batch = np.ndarray(shape=(count, image_size,image_size, self.num_channels))

    for i in range(0,count):
      [datasetIdx, imageIdx] = self._indices[start+i]
      image = self._datasets[datasetIdx].get(imageIdx)
      batch[i] = image
     
    return batch, self._labels[start:end]

  

  def nextBatch(self, batch_size, image_size = None):
    """Return the next `batch_size` examples from this data set."""
    if self._index_in_epoch >= self._num_examples:
        self._epochs_done += 1
        self._index_in_epoch = 0

    start = self._index_in_epoch


    self._index_in_epoch += batch_size
    end = self._index_in_epoch
    
    if end > self._num_examples:
        end = self._num_examples
        self._epochs_done += 1
        self._index_in_epoch = 0

    count = end-start

    batch = np.ndarray(shape=(count, image_size,image_size, self.num_channels))

    for i in range(0,count):
      [datasetIdx, imageIdx] = self._indices[self._index_in_epoch-count+i]
      image = self._datasets[datasetIdx].get(imageIdx)
      batch[i] = image
     
    return batch, self._labels[start:end]

  def nextImage(self, image_size):
    return self.next_batch(1, image_size)
    
