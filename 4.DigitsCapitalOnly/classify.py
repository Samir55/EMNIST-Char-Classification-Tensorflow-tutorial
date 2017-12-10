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

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
from os import listdir
from os.path import isfile, join, basename

import tensorflow as tf
import numpy as np
import scipy.misc as sp
import math

import loader as input_data
from PIL import Image
#import matplotlib.pyplot as plt

FLAGS = None

NUM_CLASSES = 69
DATASET_PATH = '../data/Char'
BATCH_SIZE = 10

keep_prob = tf.placeholder(tf.float32)

CHARACTERS_PATH = "../chars/"
PATH = 'out/Output1/Characters/'

def read_dict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            for x in values[1:len(values)]:
              dict[int(values[0])] = int(x)
        return(dict)

def prepare_input_character_image(path, invert):

    # Import the image
    x = Image.open(path, 'r')

    # Transform x to meet neural network specs
    x = np.rot90(np.fliplr(np.array(x)))
    
    # Invert colors
    if (invert): x = 255 - x

    # Resize to match dataset
    rows = len(x)
    cols = len(x[0])

    if (rows > cols):
      ratio = 20.0/rows
      new_dimes = 20, int(cols * ratio)
    else:
      ratio = 20.0/cols
      new_dimes = int(rows * ratio), 20
      
    x = sp.imresize(x, new_dimes, 'nearest', 'L')

    out = np.zeros((28,28), dtype=np.int)
    out[14-int(math.floor(new_dimes[0]/2.0)):14+int(math.ceil(new_dimes[0]/2.0)), 14-int(math.floor(new_dimes[1]/2.0)):14+int(math.ceil(new_dimes[1]/2.0))] = x

    y = out.reshape((28 * 28, -1)).astype(np.float32)/255.0
    return y

def get_files_list(path, remove_ext = False):
  li = list()
  for t in (listdir(path)):
    if (not isinstance(t, int) and t[0] == '.'): continue
    if (remove_ext): li.append(int(t.split('.')[0]))
    else : li.append(t)
  return li

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def main(_):
  # Import data
  #mnist = input_data.read_data_sets(DATASET_PATH, one_hot=True)
  #print ("CHECKING" , np.array(mnist.train.images).shape, np.array(mnist.train.labels).shape)
  #print ("CHECKING" , np.array(mnist.test.images).shape, np.array(mnist.test.labels).shape)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

  # Build the graph for the deep net
  y_conv = deepnn(x)

  saver = tf.train.Saver()

  # Read char mappings
  char_map = read_dict("char_map.txt", ' ')

  with tf.Session() as sess:
      # First let's load meta graph and restore weights
      saver.restore(sess, "checkpoints/model.ckpt-1")

      code = ""
      for line in sorted(np.array(get_files_list(PATH))):
        for word in sorted(np.array(get_files_list(PATH+str(line)+"/"))):
          for char in sorted(np.array(get_files_list(PATH+str(line)+"/"+str(word)+"/", True))):
            a = prepare_input_character_image(PATH+str(line)+"/"+str(word)+"/"+str(char)+".jpg", True)
            tensor = np.asarray(a)
            tensor = tensor.reshape(1, 784)
            res = np.argmax(sess.run(y_conv, {x: tensor, keep_prob: 1.0}), axis = 1)
            print (chr((char_map[res[0]])), line, " ", word)
            # code += chr((char_map[res[0]]))
            # print(code)
            # print (PATH+str(line)+"/"+str(word)+"/"+str(char)+".jpg", chr(char_map[(np.argmax(res))]), (-res).argpartition(10, axis=None)[:10])

          print (' ')
          code += ' '

        print('\n')
      code += '\n'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)