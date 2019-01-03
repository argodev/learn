# -*- coding: utf-8 -*-

import tensorflow
import hy_param

# define the placeholders to feed data into the computational graph
X = tensorflow.placeholder("float", [None, hy_param.num_input], name="input_x")
Y = tensorflow.placeholder("float", [None, hy_param.num_classes], name='input_y')

# define variables to hold the weights and biases
# we are initializing our variables with random values from a normal distribution
weights = {
    'h1': tensorflow.Variable(tensorflow.random_normal([hy_param.num_input, hy_param.n_hidden_1])),
    'h2': tensorflow.Variable(tensorflow.random_normal([hy_param.n_hidden_1, hy_param.n_hidden_2])),
    'out': tensorflow.Variable(tensorflow.random_normal([hy_param.n_hidden_2, hy_param.num_classes]))
}

biases = {
    'b1': tensorflow.Variable(tensorflow.random_normal([hy_param.n_hidden_1])),
    'b2': tensorflow.Variable(tensorflow.random_normal([hy_param.n_hidden_2])),
    'out': tensorflow.Variable(tensorflow.random_normal([hy_param.num_classes]))
}

# now, let's set up the logistic regression operation
layer_1 = tensorflow.add(tensorflow.matmul(X, weights['h1']), biases['b1'])
layer_2 = tensorflow.add(tensorflow.matmul(layer_1, weights['h2']), biases['b2'])
logits = tensorflow.matmul(layer_2, weights['out']) + biases['out']

# convert the logistic values to probablistic values using softmax (each value becomes 0..1)
prediction = tensorflow.nn.softmax(logits, name='prediction')

# define the cost function utilizing the Adam Optimizer and minimize to calculate the
# stochastic gradient descent (SGD)
loss_op = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tensorflow.train.AdamOptimizer(learning_rate=hy_param.learning_rate)
train_op = optimizer.minimize(loss_op)

# finally, let's make the prediction. These are needed to calculate and capture the accuracy values in a batch
correct_pred = tensorflow.equal(tensorflow.argmax(prediction, 1), tensorflow.argmax(Y, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_pred, tensorflow.float32), name='accuracy')
