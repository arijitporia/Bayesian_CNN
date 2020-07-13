# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 07:37:42 2019

@author: Arijit Poria
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings('ignore')
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow_probability as tfp

mnist_conv = input_data.read_data_sets('D:/PPM/',reshape=False ,one_hot=False)
mnist_conv_onehot = input_data.read_data_sets('D:/PPM/',reshape=False ,one_hot=True)

images = tf.placeholder(tf.float32,shape=[None,28,28,1])
labels = tf.placeholder(tf.float32,shape=[None,])
hold_prob = tf.placeholder(tf.float32)

# define the model
neural_net = tf.keras.Sequential([
      tfp.layers.Convolution2DReparameterization(32, kernel_size=3,  padding="SAME", activation=tf.nn.relu),
#      tf.keras.layers.MaxPooling2D(pool_size=[2, 2],  strides=[2, 2],  padding="SAME"),
      tfp.layers.Convolution2DReparameterization(64, kernel_size=3,  padding="SAME",  activation=tf.nn.relu),
#      tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
      tf.keras.layers.Flatten(),
      tfp.layers.DenseFlipout(32, activation=tf.nn.relu),
#      tf.keras.layers.Dropout(hold_prob),
      tfp.layers.DenseFlipout(10)])
logits = neural_net(images)

# Compute the -ELBO as the loss, averaged over the batch size.
labels_distribution = tfp.distributions.Categorical(logits=logits)
neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
kl = sum(neural_net.losses) / mnist_conv.train.num_examples
elbo_loss = neg_log_likelihood + kl
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(elbo_loss)

# Build metrics for evaluation. Predictions are formed from a single forward
# pass of the probabilistic layers. They are cheap but noisy predictions.
predictions = tf.argmax(logits, axis=1)
accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

learning_rate = 0.005   #initial learning rate
max_step = 1500 #number of training steps to run
batch_size = 150 #batch size
init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(max_step+1):
        images_b, labels_b = mnist_conv.train.next_batch(batch_size)
        images_h, labels_h = mnist_conv.validation.next_batch(mnist_conv.validation.num_examples)
        sess.run([train_op, accuracy_update_op], feed_dict={images: images_b,labels: labels_b,hold_prob:0.5})
        if (step==0) | (step % 1500 == 0):
            loss_value, accuracy_value = sess.run([elbo_loss, accuracy], feed_dict={images: images_b,labels: labels_b,hold_prob:0.5})
            print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(step, loss_value, accuracy_value))
    