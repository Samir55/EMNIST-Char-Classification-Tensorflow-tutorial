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

import numpy as np
import scipy.misc as sp
import math

from sklearn.externals import joblib
from PIL import Image

FLAGS = None

NUM_CLASSES = 69
DATASET_PATH = '../data/Char'
BATCH_SIZE = 10


CHARACTERS_PATH = "../chars/"
PATH = 'test_cases/Output3/'

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

# Read char mappings
char_map = read_dict("char_map.txt", ' ')

# Load model.
clf = joblib.load('models/model.pkl') 

code = ""
for line in sorted(np.array(get_files_list(PATH))):
    l = ''
    for word in sorted(np.array(get_files_list(PATH + str(line) + "/"))):
        for char in sorted(np.array(get_files_list(PATH + str(line) + "/" + str(word) + "/", True))):
            a = prepare_input_character_image(PATH + str(line) + "/" + str(word) + "/" + str(char) + ".jpg", True)
            res = clf.predict(np.array(a).reshape(1, 784))
            c = chr((char_map[res[0]]))
            code += c
            l += c
        l += ' '
        code += ' '
    print(l, '\n')
    code += '\n' 
