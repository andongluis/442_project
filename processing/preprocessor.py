'''
This will be a class that handles the preprocessing of data.
This class will preprocess data
'''
from . import dataset

class Preprocessor:

	def __init__(self, flags):
		'''
		Will initialize preprocessor

		Arguments:
		- flags: dict with bool values for all flags
				 these correspond to what of the data will we
				 use AND any settings for preprocessing
				 Missing flag will raise error
		'''


	def fit(self, filepath):
		'''
		Will learn preprocessing based off of this data

		Returns the preprocessed data as a Dataset instance

		Arguments:
		- filepath: filepath for the csv file that will have all names of all images
					along with their features and stuff
		'''


	def transform(self, filepath):
		'''
		Will preprocess the data in filepath

		Returns preprocessed data as a Dataset instance

		Arguments:
		- filepath: filepath for the csv file that will have all names of all images
					along with their features and stuff		
		'''