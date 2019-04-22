'''
Main script for doing stuff
Should be able to just call this function and set some flags
'''

import os

PATH = "./442_project/"
PATH = "/content/drive/My Drive/final_442/Code/"
os.chdir(PATH)

from model import models
from model.utility_nn import nn_train, nn_stats
from processing.preprocessor import Preprocessor
from processing.dataset import Dataset
from utility.functions import plot_history, process_raw_csv, baseline
import sys
import yaml
from sklearn.model_selection import train_test_split
from pandas import DataFrame

def main():
    # Read flags
    if len(sys.argv) > 1:
        args = [sys.argv[1] ]
        if len(sys.argv) > 2:
        	args.append(sys.argv[2])
        flags = {}
        if "images" in args:
        	flags["images"] = True
        else:
            flags["images"] = False
        if "fc" in args:
        	flags["fc"] = True
        else:
            flags["fc"] = False

        if not(flags["fc"] or flags["images"] ):
        	print("Error: no valid flags")
        	sys.exit(1)
    else:
        print("Error: need to specify flags for type of input")
        sys.exit(1)

    # Read data and have ready train, valid, test
    df = DataFrame()
    # files = ["./batch_1.csv"]
    files = ["./batch_1.csv", "./batch_3.csv", "./batch_4.csv", "./batch_5.csv", "./batch_6.csv", "./batch_7.csv", "./batch_8.csv"]
    for filename in files:
        curr_df = process_raw_csv(filename)
        df = df.append(curr_df)
    X = df.drop("likes", axis=1)
    Y = df["likes"].values

    del df
    # print(Y)
    # sys.exit(1)
    trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.20, random_state=42)
    trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.20, random_state=42)

    del X

    # Preprocess data
    processor =  Preprocessor(flags, config_file="./config/preprocess.yml")
    trainX = processor.fit(trainX, zip_file="./all_data.zip")
    validX = processor.transform(validX, zip_file="./all_data.zip")
    testX = processor.transform(testX, zip_file="./all_data.zip")

    # Build models
    multi_input = False
    if flags["fc"]:
        if flags["images"]:
            # Multi input
            multi_input = True
            fc_shape = trainX.return_fc().shape[1:]
            images_shape = trainX.return_images().shape[1:]
            model = models.multi_input_model(fc_shape, images_shape, config_file="./config/multi_input.yml",
                                             fc_file="./config/fc.yml", cnn_file="./config/cnn.yml")
        else:
            # Only fc
            trainX = trainX.return_fc()
            validX = validX.return_fc()
            testX = testX.return_fc()
            model = models.fc_model(trainX.shape[1:], config_file="./config/fc.yml")
            print(trainX.shape)
            print(validX.shape)
    else:
        # Only images
        # Something about images
        trainX = trainX.return_images()
        validX = validX.return_images()
        testX = testX.return_images()
        print(trainX.shape[1:])
        model = models.cnn_model(trainX.shape[1:],  config_file="./config/cnn.yml")

    # Train models
    model, adam_hist, sgd_hist = nn_train(model, trainX, trainY, validX, validY, multi_input=multi_input, config_file="./config/train.yml")    

    # Give results
    y_mean = baseline(trainY, name="Train")
    nn_stats(model, trainX, trainY, multi_input=multi_input, name="Train")
    _ = baseline(validY, name="Valid")
    nn_stats(model, validX, validY, multi_input=multi_input, name="Valid")
    _ = baseline(testY, y_mean, name="Test")
    nn_stats(model, testX,  testY,  multi_input=multi_input, name="Test")

    # Plot stuff
    plot_history(adam_hist, sgd_hist, path="./", acc=True, loss=True)


if __name__ == "__main__":
    main()