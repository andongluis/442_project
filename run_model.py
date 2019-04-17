'''
Main script for doing stuff
Should be able to just call this function and set some flags
'''
from model import models
from model.utility_nn import nn_train, nn_stats
from processing.preprocessor import Preprocessor
from processing.dataset import Dataset
from utility.functions import plot_history, process_raw_csv
import sys
import yaml
from sklearn.model_selection import train_test_split


def main():
    # Read flags
    if len(sys.argv) > 1:
        args = [sys.argv[1] ]
        if len(sys.argv) > 2:
        	args.append(sys.argv[2])
        flags = {}
        if "images" in args:
        	flags["images"] = True
        if "fc" in args:
        	flags["fc"] = True

        if len(flags.keys()) == 0:
        	print("Error: no valid flags")
        	sys.exit(1)
    else:
        print("Error: need to specify flags for type of input")
        sys.exit(1)


    # Read config files


    # Read data and have ready train, valid, test
    df = process_raw_csv("batch_1.csv")
    X = df.drop("likes", axis=1)
    Y = df["likes"]
    trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.25)
    trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.25)

    # Preprocess data
    processor =  Preprocessor(flags, config_file="./config/preprocess.yml")
    train_D = processor.fit(trainX)
    valid_D = processor.transform(validX)
    test_D = processor.transform(testX)



    # Build models
    model = models.cnn_model(trainX.shape)

    # Train models
    model, adam_hist, sgd_hist = nn_train(model, trainX, trainY, validX, validY, multi_input=False)

    # Give results
    nn_stats(model, trainX, trainY, multi_input=False, name="Train")
    nn_stats(model, validX, validY, multi_input=False, name="Valid")
    nn_stats(model, testX,  testY,  multi_input=False, name="Test")

    # Plot stuff
    plot_history(adam_hist, sgd_hist, path="./", acc=True, loss=True)


if __name__ == "__main__":
    main()