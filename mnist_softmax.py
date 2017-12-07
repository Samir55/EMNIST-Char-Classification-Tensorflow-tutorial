# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import mnist as input_data
import numpy as np
import tensorflow as tf

# ToDo @Samir55 add the number of epoches / Fix reduce entropy

FLAGS = None

NUM_CLASSES = 62
DATASET_PATH = 'data/Char'

BATCH_SIZE = 10

def main(_):
  # Import data
  mnist = input_data.read_data_sets(DATASET_PATH, one_hot=True)

  # Create the model
  # Add the first layer.
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 300]))
  b = tf.Variable(tf.zeros([300]))
  y = tf.matmul(x, W) + b

  # Add the second layer.
  W2 = tf.Variable(tf.zeros([300, 62]))
  b2 = tf.Variable(tf.zeros([62]))
  y2 = tf.matmul(y, W2) + b2  

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 62])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y2))
  train_step = tf.train.GradientDescentOptimizer(2.0).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  number_of_batches = int(len(mnist.train.images) / BATCH_SIZE)
  for _ in range(number_of_batches):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  # Test a character image.
  a = mnist.train.next_batch(1)
  np.set_printoptions(linewidth=250)
  char_image = a[0]

  # Reshape into 28x28 from 1x784
  char_image = np.array(char_image.reshape(28, 28)) * int(255)
  char_image = np.array(char_image, dtype=int)

  # Inverse Transformations
  char_image = np.flip(char_image, axis = 0) # Inverse flip at axis 0.
  char_image = np.rot90(char_image,3) # Rotate 270 counter clock-wise

  # Reshape into 28x28 from 1x784
  char_image = np.array(char_image.reshape(28, 28)) * int(255)
  char_image = np.array(char_image, dtype=int)
  
  print ("Image is " , char_image , "Correct Label is", np.argmax(a[1], axis =  1))
  print ("Predicted value for image 0 is ", np.argmax(sess.run(y2, {x: a[0]}), axis = 1))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
