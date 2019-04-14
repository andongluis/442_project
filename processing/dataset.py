'''
Class for the dataset
Can return various parts of the dataset depending
on what you need
'''


class Dataset:

	def __init__(self, flags):
		'''
		Initializes flags
		Potential flags:
		- "images": True/False
		- "fc": True/False
		'''

	def set_images(self, images):
		'''
		Sets image data.
		Should return error if flag indicates no images
		'''

	def set_fc_data(self, fc_data):
		'''
		Sets fully-connected data.
		Should return error if flag indicates no fully-connected
		'''

	def return_images(self):
		'''
		Returns image data.
		Should return error if flag indicates no images
		'''

	def return_fc_data(self):
		'''
		Returns fully-connected data.
		Should return error if flag indicates no fully-connected
		'''
