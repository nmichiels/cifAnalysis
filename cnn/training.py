from cifDataset.cifStreamer.dataset import Dataset
import numpy as np
import tensorflow as tf
from cnn.cnn import create_convolutional_layer, create_fc_layer, create_flatten_layer
import time



def do_eval(sess,
            eval_correct, loss, images_pl, labels_pl,
            dataset, img_size, channels, batch_size):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    total_loss = 0
    steps_per_epoch = dataset.num_examples // batch_size + 1
        
    # num_examples = steps_per_epoch * FLAGS.batch_size
    num_examples = 0
    for step in range(steps_per_epoch):
        x_batch, labels_batch = dataset.next_batch(batch_size, img_size)
        x_batch = x_batch[:,:,:,channels]
        feed_dict={images_pl: x_batch, labels_pl: labels_batch}

        [count, loss_value] = sess.run([eval_correct,loss], feed_dict=feed_dict)
        true_count += count
        total_loss +=loss_value
        num_examples += x_batch.shape[0]

    precision = float(true_count) / num_examples
    mean_loss = total_loss / steps_per_epoch
    return num_examples, true_count, precision, mean_loss
    # print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            # (num_examples, true_count, precision))

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))  
        # Do the actual training
        
            
def train(  data,
            classes,
            channels,
            max_iterations,
            max_epochs,
            img_size,
            outputModel,
            batch_size = 256,
            permutate = True,
            masking = False,
            validation_size = 0.2,
            display_epoch = 1,
            dropout = 0.25,
            logs_path = './tmp/tensorflow_logs/example/'
        ):
    tf.set_random_seed(2)

    print("\nStart training...")
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

    num_classes = data.train.num_classes
    num_channels = len(channels)

    # Constructing CNN
    # --------------------------------
    session = tf.Session()
    x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
    mask = tf.placeholder(tf.bool, shape=[None, img_size,img_size,num_channels], name='mask')

    ## labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)

    #Network graph params
    filter_size_conv1 = 3 
    num_filters_conv1 = 32
    filter_size_conv2 = 3
    num_filters_conv2 = 32
    filter_size_conv3 = 3
    num_filters_conv3 = 64
    fc_layer_size = 128


    # # masking
    # images = x
    # if masking:
    #     images = tf.boolean_mask(x, mask,  name='boolean_mask',    axis=None)


    layer_conv1 = create_convolutional_layer(input=x,
                num_input_channels=num_channels,
                conv_filter_size=filter_size_conv1,
                num_filters=num_filters_conv1, 
                layerName="hidden_layer_1")
    layer_conv2 = create_convolutional_layer(input=layer_conv1, 
                num_input_channels=num_filters_conv1,
                conv_filter_size=filter_size_conv2,
                num_filters=num_filters_conv2, 
                layerName= "hidden_layer_2")

    layer_conv3= create_convolutional_layer(input=layer_conv2,
                num_input_channels=num_filters_conv2,
                conv_filter_size=filter_size_conv3,
                num_filters=num_filters_conv3, 
                layerName = "hidden_layer_3")
          
            
    layer_flat = create_flatten_layer(layer_conv3, layerName = "flatten_layer_1")

    layer_fc1 = create_fc_layer(input=layer_flat,
                        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                        num_outputs=fc_layer_size,
                        layerName = "fc1",
                        #keep_prob=keep_prob,
                        use_relu=True)

    layer_fc2 = create_fc_layer(input=layer_fc1,
                        num_inputs=fc_layer_size,
                        num_outputs=num_classes,
                        layerName = "fc1",
                        #keep_prob=keep_prob,
                        use_relu=False) 

    y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

    y_pred_cls = tf.argmax(y_pred, axis=1)
    session.run(tf.global_variables_initializer())
    

    # define optimization functions (training)
    #  ---------------------------------------------------
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.07)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
    # loss = tf.reduce_mean(cross_entropy)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_true_cls, logits=layer_fc2) 
    train_op = optimizer.minimize(loss=loss)

     # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(y_pred, y_true_cls)
    # eval_correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    # eval_correct = tf.reduce_sum(tf.cast(eval_correct_prediction, tf.float32))
    # eval_correct = tf.reduce_mean(tf.cast(eval_correct, tf.float32))

    session.run(tf.global_variables_initializer()) 

    saver = tf.train.Saver()

    def train_epochs(num_epochs):
        total_iterations = 0

        for epoch in range(num_epochs):
            start_time = time.time()


            # Loop over all batches and train network
            total_batch = data.train.num_examples // batch_size + 1 
            for i in range(total_batch):
                x_batch, y_true_batch = data.train.next_batch(batch_size, img_size)
                x_batch = x_batch[:,:,:,channels]
                x_valid_batch, y_valid_batch = data.test.next_batch(batch_size, img_size)
                x_valid_batch = x_valid_batch[:,:,:,channels]
                feed_dict_tr = {x: x_batch, y_true: y_true_batch}
                # _, loss_value = session.run([train_op, loss], feed_dict=feed_dict_tr)
                session.run([train_op], feed_dict=feed_dict_tr)
                # summary_writer.add_summary(summary, epoch * total_batch + i)


            # Evaluate against the training set.
            [num_examples_train, true_count_train, precision_train, mean_loss_train] = do_eval(session, eval_correct, loss, x, y_true, data.train, img_size, channels, batch_size)
            # Evaluate against the validation set.
            [num_examples_valid, true_count_valid, precision_valid, mean_loss_valid] = do_eval(session, eval_correct, loss, x, y_true,  data.test, img_size, channels, batch_size)

            duration = time.time() - start_time
            print('Epoch %3d: --- Training Accuracy: %5.2f%%, loss:%1.5f (%5d/%5d) -- Validation Accuracy: %5.2f%%, loss:%1.5f (%5d/%5d) [%.3f sec]' % (epoch+1, precision_train*100., mean_loss_train, true_count_train, num_examples_train, precision_valid*100.,  mean_loss_valid, true_count_valid, num_examples_valid, duration))

            if (permutate):
                data.train.permutate()

            saver.save(session, outputModel) 

            total_iterations = total_iterations + total_batch

        print("Optimization Finished with ", total_iterations, " iterations and ", num_epochs, " epochs!")



    print("Run the command line:\n" \
            "--> tensorboard --logdir=/tmp/tensorflow_logs " \
            "\nThen open http://0.0.0.0:6006/ into your web browser")
    # train_iterations(num_iteration=max_iterations, max_epochs=max_epochs)
    train_epochs(num_epochs=max_epochs)

