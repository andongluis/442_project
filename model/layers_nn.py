'''
Has common "layers"/blocks/sets for neural networks
'''

import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import PReLU


def cnn_set(in_tensor, filters, kernel_size, drop_rate, act, b_norm, pool, pool_size):
    if act != "prelu":
        layer = Conv2D(filters, kernel_size, strides=(1, 1), padding='same', activation=act)(in_tensor)
    else:
        layer = Conv2D(filters, kernel_size, strides=(1, 1), padding='same')(in_tensor)
    if b_norm:
        layer = BatchNormalization()(layer)
    if act == "prelu":
        layer = PReLU()(layer)
    if pool:
        layer = MaxPooling2D(pool_size=pool_size, padding='same')(layer)
    layer = Dropout(rate=drop_rate)(layer)
    return layer

def fc_set(in_tensor, fc_cells, drop_rate, act, b_norm):
    if act != "prelu":
        layer = Dense(fc_cells, activation=act)(in_tensor)
    else:
        layer = Dense(fc_cells, activation=None)(in_tensor)
    if b_norm:
        layer = BatchNormalization()(layer)
    if act == "prelu":
        layer = PReLU()(layer)
    layer = Dropout(rate=drop_rate)(layer)
    return layer    
