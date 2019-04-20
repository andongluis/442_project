'''
Has functions that set up training for the model
'''
import numpy as np

from keras.losses import mean_absolute_error
from keras.optimizers import SGD
from keras.backend import eval

from keras import callbacks
import yaml

def generator(trainX, trainY, batch_size=64):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, trainX.shape[1]))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        indices = np.random.choice(trainX.shape[0], batch_size)
        batchX = trainX[indices]
        batchY = trainY[indices]
        yield batchX, batchY


def multi_input_generator(trainX, trainY, batch_size):
    pass


def nn_stats(clf, trainX, trainY, multi_input=False, name="Train"):

    if multi_input:
        y_pred = clf.predict(trainX)
    else:
        y_pred = clf.predict(trainX)

    train_mae = eval(mean_absolute_error(trainY, y_pred)).mean()
    print("{} performance average MAE: {}".format(name, round(train_mae, 2)) )


def nn_train(model, trainX, trainY, validX, validY, multi_input=False, config_file="config/train.yml"):
    # Note: trainX, validX should be a dataset object if multi_input is true.
    # Otherwise, they should be regular matrices/df

    print("--------Training Model---------")

    with open(config_file, 'r') as file:
        PARAMS = yaml.load(file, Loader=yaml.FullLoader)

    iterations = PARAMS["iterations"]
    sgd_iter = PARAMS["sgd_iter"]
    lr = PARAMS["sgd_lr"]
    momentum = PARAMS["sgd_momentum"]
    decay = PARAMS["sgd_decay"]
    nesterov = PARAMS["sgd_nesterov"]

    steps_per_epoch = PARAMS["steps_per_epoch"]
    validation_steps = PARAMS["validation_steps"]
    batch_size = PARAMS["batch_size"]

    # Fit and train
    callbacks_list = [callbacks.TerminateOnNaN(),]

    model.compile(optimizer="adam", loss=mean_absolute_error)
    if multi_input:
        # adam_hist = model.fit_generator(multi_input_generator(trainX, trainY, batch_size), steps_per_epoch=steps_per_epoch,
        #                                 epochs=iterations, verbose=2, 
        #                                 callbacks=callbacks_list, validation_data=multi_input_generator(validX, validY, batch_size),
        #                                 validation_steps=validation_steps, class_weight=None, max_queue_size=10, 
        #                                 workers=4, use_multiprocessing=True, shuffle=True, initial_epoch=0)

        print("Gonna change to sgd now")

        sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)

        model.compile(optimizer=sgd, loss=mean_absolute_error)

        # sgd_hist = model.fit_generator(multi_input_generator(trainX, trainY, batch_size), steps_per_epoch=steps_per_epoch, 
        #                                 epochs=sgd_iter, verbose=2, 
        #                                 callbacks=callbacks_list, validation_data=multi_input_generator(validX, validY, batch_size),
        #                                 validation_steps=validation_steps, class_weight=None, max_queue_size=10, 
        #                                 workers=4, use_multiprocessing=True, shuffle=True, initial_epoch=0)
    else:
        # adam_hist = model.fit_generator(generator(trainX, trainY, batch_size), steps_per_epoch=steps_per_epoch,
        #                                 epochs=iterations, verbose=2, 
        #                                 callbacks=callbacks_list, validation_data=generator(validX, validY, batch_size),
        #                                 validation_steps=validation_steps, class_weight=None, max_queue_size=10, 
        #                                 workers=4, use_multiprocessing=True, shuffle=True, initial_epoch=0)
        adam_hist = model.fit(trainX, trainY,  batch_size=batch_size, epochs=iterations,
                             verbose=1, callbacks=callbacks_list, validation_data=(validX, validY))
        print("Gonna change to sgd now")

        sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)

        model.compile(optimizer=sgd, loss=mean_absolute_error)

        sgd_hist = model.fit(trainX, trainY,  batch_size=batch_size, epochs=sgd_iter,
                             verbose=1, callbacks=callbacks_list, validation_data=(validX, validY))

        # sgd_hist = model.fit_generator(generator(trainX, trainY, batch_size), steps_per_epoch=steps_per_epoch, 
        #                                 epochs=sgd_iter, verbose=2, 
        #                                 callbacks=callbacks_list, validation_data=generator(validX, validY, batch_size),
        #                                 validation_steps=validation_steps, class_weight=None, max_queue_size=10, 
        #                                 workers=4, use_multiprocessing=True, shuffle=True, initial_epoch=0)
    return model, adam_hist, sgd_hist