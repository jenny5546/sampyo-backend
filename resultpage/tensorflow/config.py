## Built-in packages
import getopt
import json
import os
import sys

## Third-party packages
from PIL import Image
import joblib
import numpy as np
import tqdm

## Tensorflow
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import SeparableConv2D
import tensorflow_addons as tfa

## Global variable declarations
global INPUT_WIDTH
global INPUT_HEIGHT
global FILTER_SIZE
global DENSE_UNITS
global DROPOUT
global OUTPUT_CLASS

## Global model parameters (DO NOT CHANGE)
INPUT_WIDTH = 1500
INPUT_HEIGHT = 850
FILTER_SIZE = 32
DENSE_UNITS = 1024
DROPOUT = 0.3
OUTPUT_CLASS = 3
