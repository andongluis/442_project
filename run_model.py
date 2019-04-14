'''
Main script for doing stuff
Should be able to just call this function and set some flags
'''
from model import models
from model.utility_nn import nn_train, nn_stats
from processing import Preprocessor, Dataset
from utility.graphing import plot_history
import sys
import yaml

# Read flags
if len(sys.argv)> 1:
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


# Read data and preprocess it
# Have ready train, valid, test
trainX, trainY
validX, validY
testX, testY

Preprocessor()


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