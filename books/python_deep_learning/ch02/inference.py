# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import hy_param

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# pointing to the model checkpoint
checkpoint_file = tf.train.latest_checkpoint(os.path.join(hy_param.checkpoint_dir, 'checkpoints'))
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

# Loading test data
test_data = np.array([mnist.test.images[8]])

# loading input variable from the model
input_x = tf.get_default_graph().get_operation_by_name("input_x").outputs[0]

# loading prediction operation
prediction = tf.get_default_graph().get_operation_by_name('prediction').outputs[0]

with tf.Session() as sess:
    # restoring the model from the checkpoint
    saver.restore(sess, checkpoint_file)

    # executing the model to make predictions
    data = sess.run(prediction, feed_dict={ input_x: test_data })
    print("Predicted digit: ", data.argmax())

    # display the feed image
    print("Input image:")
    plt.gray()
    plt.imshow(test_data.reshape([28,28]))
    plt.show()