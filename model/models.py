'''
Model-building functions
'''

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2

import yaml


def cnn_model(in_shape, config_file="config/cnn.yml"):
	# CNN model, no supplemental data
	with open(config_file, 'r') as file:
	    PARAMS = yaml.load(file)

	drop_rate = PARAMS["drop_rate"]
	filters = PARAMS["filters"]
	kernel_size = PARAMS["kernel_size"]
	act = PARAMS["act"]
	b_norm = PARAMS ["b_norm"]
	cnn_depth = PARAMS["cnn_depth"]
	fc_depth = PARAMS["fc_depth"]
	fc_cells = PARAMS["fc_cells"]
	pool = PARAMS["pool"]
	pool_size = PARAMS["pool_size"]

	in_tensor = Input(shape=in_shape)
	layer = cnn_set(in_tensor, filters, kernel_size, drop_rate, act, b_norm, pool, pool_size)
	for i in range(cnn_depth - 1):
		layer = cnn_set(layer, filters, kernel_size, drop_rate, act, b_norm, pool, pool_size)

	# flatten
	for i in range(fc_depth):
		layer = fc_set(layer, fc_cells, drop_rate, act, b_norm)

	output = Dense(1, activation=None)(layer)
	return Model(inputs=in_tensor, outputs=output)

def fc_model(in_shape, config_file="config/config.yml"):
	# FC model, no images
	with open(config_file, 'r') as file:
	    PARAMS = yaml.load(file)

	drop_rate = PARAMS["drop_rate"]
	act = PARAMS["act"]
	b_norm = PARAMS ["b_norm"]
	fc_depth = PARAMS["fc_depth"]
	fc_cells = PARAMS["fc_cells"]

	in_tensor = Input(shape=in_shape)
	layer = fc_set(in_tensor, fc_cells drop_rate, act, b_norm)
	for i in range(fc_depth - 1):
		layer = fc_set(layer, fc_cells, drop_rate, act, b_norm)

	output = Dense(1, activation=None)(layer)
	return Model(inputs=in_tensor, outputs=output)


def multi_input_model(TBD):
	# FC and CNN model, images and suplemental data
	pass