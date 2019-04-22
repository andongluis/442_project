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
from keras.layers import Flatten
from keras.layers import Concatenate

from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2

from .layers_nn import *

import yaml


def cnn_model(in_shape, config_file="config/cnn.yml", out_cells=1):
    # CNN model, no supplemental data
    with open(config_file, 'r') as file:
        PARAMS = yaml.load(file, Loader=yaml.FullLoader)

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
    layer = Flatten()(layer)
    for i in range(fc_depth):
        layer = fc_set(layer, fc_cells, drop_rate, act, b_norm)

    output = Dense(out_cells, activation=None)(layer)
    return Model(inputs=in_tensor, outputs=output)

def fc_model(in_shape, config_file="config/fc.yml", out_cells=1):
    # FC model, no images
    with open(config_file, 'r') as file:
        PARAMS = yaml.load(file, Loader=yaml.FullLoader)

    drop_rate = PARAMS["drop_rate"]
    act = PARAMS["act"]
    b_norm = PARAMS ["b_norm"]
    fc_depth = PARAMS["fc_depth"]
    fc_cells = PARAMS["fc_cells"]

    in_tensor = Input(shape=in_shape)
    layer = fc_set(in_tensor, fc_cells, drop_rate, act, b_norm)
    for i in range(fc_depth - 1):
        layer = fc_set(layer, fc_cells, drop_rate, act, b_norm)

    output = Dense(out_cells, activation=None)(layer)
    return Model(inputs=in_tensor, outputs=output)


def multi_input_model(fc_in_shape, images_in_shape, config_file="config/multi_input.yml",
                      fc_file="config/fc.yml", cnn_file="config/cnn.yml"):
    # FC and CNN model, images and supplemental data

    with open(config_file, 'r') as file:
        PARAMS = yaml.load(file, Loader=yaml.FullLoader)

    drop_rate = PARAMS["drop_rate"]
    act = PARAMS["act"]
    b_norm = PARAMS ["b_norm"]
    multi_depth = PARAMS["fc_depth"]
    fc_cells = PARAMS["fc_cells"]
    out_cells = PARAMS["out_cells"]

    fc = fc_model(fc_in_shape, config_file=fc_file, out_cells=out_cells)
    cnn = cnn_model(images_in_shape, config_file=cnn_file, out_cells=out_cells)

    layer = Concatenate()([fc.output, cnn.output])

    for i in range(multi_depth):
        layer = fc_set(layer, fc_cells, drop_rate, act, b_norm)

    output = Dense(1, activation=None)(layer)
    return Model(inputs=[fc.input, cnn.input], outputs=output)

def bagodwords(in_shape, config_file="config/fc.yml", out_cells=1):
    with open(config_file, 'r') as file:
        PARAMS = yaml.load(file, Loader=yaml.FullLoader)
    
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
    tokenize = text.Tokenizer(num_words=50)
    tokenize.fit_on_texts(train_posts)
    x_train = tokenize.texts_to_matrix(train_posts)
    encoder = LabelBinarizer()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)
    y_test = encoder.transform(test_tags)
    in_tensor = Sequential()
    in_tensor.add(Dense(512, input_shape=(in_shape,)))
    in_tensor.add(Activation('relu'))

    in_tensor.add(Dense(num_labels))
    in_tensor.add(Activation('softmax'))
    in_tensor.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = in_tensor.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=20,
                        verbose=1,
                        validation_split=0.1)

    score = in_tensor.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
