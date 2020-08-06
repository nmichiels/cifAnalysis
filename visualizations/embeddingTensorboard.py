import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector



def extractActivationFeatures( sess, graph, data, classes, channels, numBatches, batch_size = 256):
    

    x= graph.get_tensor_by_name("x:0") 
    activationfeatures = graph.get_tensor_by_name("flatten_layer_1:0")


    numError = 0
    total = 0
    total_batch = int(data.test.num_examples/batch_size)


    # np.empty((data.test.num_classes, 0, data.img_size, data.img_size, data.test.num_channels))
    features = np.empty(shape=(0, 1600))

    for i in range(numBatches):
        x_batch_full, y_true_batch = data.test.next_batch(batch_size)
        x_batch = x_batch_full[:,:,:,channels]


        feed_dict_embedding = {x: x_batch}
        features_batch = sess.run(activationfeatures, feed_dict=feed_dict_embedding)
        # print(features_batch.shape)
        features = np.append(features, features_batch, axis=0)

    print(features.shape)
    return features




   
# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def embedding(data, classes, channels, feature_vectors, logdir, name, img_size = 35):
    num_of_samples=feature_vectors.shape[0]

    x, y = data.test.nextBatch(num_of_samples,img_size)

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