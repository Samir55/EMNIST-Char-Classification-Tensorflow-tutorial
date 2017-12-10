# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
import numpy as np
import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

from os import listdir
from os.path import isfile, join
import numbers

import PIL.ImageOps    
from PIL import Image  

size = (20,20)
new_size = (28, 28)
CHARACTERS_PATH = '/home/ahmedsamir/Desktop/NN/org_set/'
SYMBOLS_PATH = '../data/Symbols/'
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
NUM_CLASSES = 69

char_map = {}

def load_capital():
  images = [] 
  labels = []

  # Read digits and capital.
  for i in range(0, 36):
    for img in listdir((CHARACTERS_PATH+str(i))):
      if (img[0] == '.'): continue
      x = Image.open((CHARACTERS_PATH + str(i) + '/' + img), 'r')
      images.append(255-np.array(x))
      #a = np.zeros(NUM_CLASSES)
      char_map[i] = i
      #a[char_map[i]] = 1
      labels.append(char_map[i])

  return images, labels

def add_chars(training_set, training_results, test_set, test_results):
  # Add general sybmols
  training_set, training_results, test_set, test_results = add_special_chars_to_training_set(training_set, training_results, test_set, test_results)
  
  # Rotate > && < to get ^
  cnt = len(listdir((SYMBOLS_PATH + 'ls')))
  i = 0
  for img in listdir((SYMBOLS_PATH + 'ls')):
    if img == '.DS_Store': continue
    # Read the image, invert it, resize it to 20*20
    x = Image.open((SYMBOLS_PATH + 'ls' + '/' + img), 'r')
    x = PIL.ImageOps.invert(x)
    x.thumbnail(size, Image.ANTIALIAS)

    # Add the frame -> new image 28*28
    new_im = Image.new("L", new_size)
    new_im.paste(x, (int((new_size[0]-size[0])/2), int((new_size[1]-size[1])/2)))

    img = np.array(new_im)
    img = np.rot90(np.rot90(np.rot90(img)))
    # Apply transformations to add to the traning set
    img = np.rot90(np.fliplr(img))

    if (i < (5.0/6) * cnt):
      # Append to dataset
      training_set.append(img)
      char_map[87] = 68
      #a = np.zeros(NUM_CLASSES)
      #a[char_map[87]] = 1
      training_results.append(char_map[87])
    else:
      # Append to testset
      test_set.append(img)
      #a = np.zeros(NUM_CLASSES)
      #a[char_map[87]] = 1
      test_results.append(char_map[87])

    i+=1
  
  cnt = len(listdir((SYMBOLS_PATH + 'gt')))
  i = 0
  for img in listdir((SYMBOLS_PATH + 'gt')):
    if img == '.DS_Store': continue

    x = Image.open((SYMBOLS_PATH + 'gt' + '/' + img), 'r')
    x = PIL.ImageOps.invert(x)
    x.thumbnail(size, Image.ANTIALIAS)

    new_im = Image.new("L", new_size)
    new_im.paste(x, (int((new_size[0]-size[0])/2), int((new_size[1]-size[1])/2)))

    img = np.array(new_im)
    img = np.rot90(img)

    img = np.rot90(np.fliplr(img))

    if (i < (5.0/6) * cnt):
      training_set.append(img)
      #a = np.zeros(NUM_CLASSES)
      #a[char_map[87]] = 1
      training_results.append(char_map[87])
    else:
      test_set.append(img)
      #a = np.zeros(NUM_CLASSES)
      #a[char_map[87]] = 1
      test_results.append(char_map[87])
    i+=1
    
  cnt = len(listdir((SYMBOLS_PATH + 'dash')))
  i = 0
  for img in listdir((SYMBOLS_PATH + 'dash')):
    if img == '.DS_Store': continue

    # Read the image, invert it, resize it to 20*20
    x = Image.open((SYMBOLS_PATH + 'dash' + '/' + img), 'r')
    x = PIL.ImageOps.invert(x)
    x.thumbnail(size, Image.ANTIALIAS)

    new_x = Image.new("L", size)
    new_x.paste(x, (0, 7))

    # Add the frame -> new image 28*28
    new_im = Image.new("L", new_size)
    new_im.paste(new_x, (int((new_size[0]-size[0])/2), int((new_size[1]-size[1])/2)))

    #print np.array(new_im)
    img = np.array(new_im)

    # Apply transformations to add to the traning set
    img = np.rot90(np.fliplr(img))

    if (i < (5.0/6) * cnt):
      # Append to dataset
      training_set.append(img)
      char_map[88] = 67
      #a = np.zeros(NUM_CLASSES)
      #a[char_map[88]] = 1
      training_results.append(char_map[88])
    else:
      # Append to testset
      test_set.append(img)
      #a = np.zeros(NUM_CLASSES)
      #a[char_map[88]] = 1
      test_results.append(char_map[88])
    i+=1
  return training_set, training_results, test_set, test_results

def add_special_chars_to_training_set(training_set, training_results, test_set, test_results):
  # Loop over all symbols in the folder -> Loop over all symbol images
  for symbol in listdir(SYMBOLS_PATH):
    if symbol == '.DS_Store': continue

    cnt = len(listdir((SYMBOLS_PATH + symbol)))
    i = 0
    for img in listdir((SYMBOLS_PATH + symbol)):
      if img == '.DS_Store': continue
      
      # Read the image, invert it, resize it to 20*20
      x = Image.open(((SYMBOLS_PATH + str(symbol)) + '/' + img), 'r')
      x = PIL.ImageOps.invert(x)

      # Resize and add frame if not ready already
      if (symbol == 'bslash' or symbol == 'dash' or symbol == 'eq' or symbol == 'exclamation' or symbol == 'forwards' or symbol == 'gt'or symbol == 'lb'or symbol == 'lcb'or symbol == 'ls'or symbol == 'lsb'or symbol == 'or'or symbol == 'plus'or symbol == 'rb'or symbol == 'rcb'or symbol == 'rsb'):
        x.thumbnail(size, Image.ANTIALIAS)

        # Add the frame -> new image 28*28
        new_im = Image.new("L", new_size)
        new_im.paste(x, (int((new_size[0]-size[0])/2), int((new_size[1]-size[1])/2)))
      else: new_im = x

      img = np.array(new_im)

      # Apply transformations to add to the traning set
      img = np.rot90(np.fliplr(img))

      if (symbol == 'forwards'):
        symbolKey = 76
        char_map[symbolKey] = 36
      elif (symbol == 'colons'):
        symbolKey = 77
        char_map[symbolKey] = 37
      elif (symbol == 'dots'):
        symbolKey = 75
        char_map[symbolKey] = 38
      elif (symbol == 'and'):
        symbolKey = 67
        char_map[symbolKey] = 39
      elif (symbol == 'or'):
        symbolKey = 91
        char_map[symbolKey] = 40
      elif (symbol == 'at'):
        symbolKey = 83
        char_map[symbolKey] = 41
      elif (symbol == 'hash'):
        symbolKey = 64
        char_map[symbolKey] = 42
      elif (symbol == 'mod'):
        symbolKey = 66
        char_map[symbolKey] = 43
      elif (symbol == 'bslash'):
        symbolKey = 85
        char_map[symbolKey] = 44
      elif (symbol == 'btick'):
        symbolKey = 89
        char_map[symbolKey] = 45
      elif (symbol == 'comma'):
        symbolKey = 73
        char_map[symbolKey] = 46
      elif (symbol == 'dash'):
        symbolKey = 74
        char_map[symbolKey] = 47
      elif (symbol == 'dollar'):
        symbolKey = 65
        char_map[symbolKey] = 48
      elif (symbol == 'dquote'):
        symbolKey = 63
        char_map[symbolKey] = 49
      elif (symbol == 'quote'):
        symbolKey = 68
        char_map[symbolKey] = 50
      elif (symbol == 'eq'):
        symbolKey = 80
        char_map[symbolKey] = 51
      elif (symbol == 'exclamation'):
        symbolKey = 62
        char_map[symbolKey] = 52
      elif (symbol == 'gt'):
        symbolKey = 81
        char_map[symbolKey] = 53
      elif (symbol == 'lb'):
        symbolKey = 69
        char_map[symbolKey] = 54
      elif (symbol == 'rb'):
        symbolKey = 70
        char_map[symbolKey] = 55
      elif (symbol == 'lcb'):
        symbolKey = 90
        char_map[symbolKey] = 56
      elif (symbol == 'rcb'):
        symbolKey = 92
        char_map[symbolKey] = 57
      elif (symbol == 'rsb'):
        symbolKey = 86
        char_map[symbolKey] = 58
      elif (symbol == 'lsb'):
        symbolKey = 84
        char_map[symbolKey] = 59
      elif (symbol == 'ls'):
        symbolKey = 79
        char_map[symbolKey] = 60
      elif (symbol == 'multi'):
        symbolKey = 71
        char_map[symbolKey] = 61
      elif (symbol == 'plus'):
        symbolKey = 72
        char_map[symbolKey] = 62
      elif (symbol == 'question'):
        symbolKey = 82
        char_map[symbolKey] = 63
      elif (symbol == 'semicolon'):
        symbolKey = 78
        char_map[symbolKey] = 64
      elif (symbol == 'tilde'):
        symbolKey = 93
        char_map[symbolKey] = 65
      else:
        symbolKey = (ord(symbol))
        char_map[symbolKey] = 66

      if (i < (5.0/6) * cnt):
        training_set.append(img)
        #a = np.zeros(NUM_CLASSES)
        #a[char_map[symbolKey]] = 1
        training_results.append(char_map[symbolKey])
      else:
        # Append to testset
        test_set.append(img)
        #a = np.zeros(NUM_CLASSES)
        #a[char_map[symbolKey]] = 1
        test_results.append(char_map[symbolKey])
      i+=1
  
  return training_set, training_results, test_set, test_results

def dense_to_one_hot(labels_dense, num_classes = NUM_CLASSES):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    self._images = np.array(images)
    self._images = self._images.astype(numpy.float32)
    self._images = numpy.multiply(self._images, 1.0 / 255.0)
    self._num_examples = self._images.shape[0]
    print("NUMBER OF EXAMPLES ", self._num_examples)
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir = "",
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  trX, trY = load_capital()

  train_images, train_labels , test_images, test_labels = add_chars(trX, trY, [], [])

  train_len = len(train_images)
  test_len = len(test_images)

  train_images = np.array(train_images).reshape([train_len, 784])
  test_images = np.array(test_images).reshape([test_len, 784])

  #train_labels = np.array(train_labels).reshape([train_len, NUM_CLASSES])
  #test_labels = np.array(test_labels).reshape([test_len, NUM_CLASSES])
  

  # Convert labels to one_hot vectors.
  train_labels = np.array(train_labels)
  test_labels = np.array(test_labels)

  print  (len(test_labels), np.array(test_images).shape, np.array(train_images).shape, np.array(test_labels).shape, np.array(test_labels).shape[0])
  #test_labels = dense_to_one_hot(test_labels, NUM_CLASSES)
  #train_labels = dense_to_one_hot(train_labels, NUM_CLASSES)

  options = dict(dtype=dtype, reshape=reshape, seed=seed)

  train = DataSet(train_images, train_labels, **options)
  validation = DataSet([], [], **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)
