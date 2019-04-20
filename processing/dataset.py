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
        self.flags = flags

    def set_images(self, images_data):
        '''
        Sets image data.
        Should return error if flag indicates no images
        '''
        if self.flags["images"]:
            self.images_data = images_data
        else:
            raise Error("images is not set for this Dataset object")

    def set_fc_data(self, fc_data):
        '''
        Sets fully-connected data.
        Should return error if flag indicates no fully-connected
        '''
        if self.flags["fc"]:
            self.fc_data = fc_data
        else:
            raise Error("fc is not set for this Dataset object")

    def return_images(self):
        '''
        Returns image data.
        Should return error if flag indicates no images
        '''
        if self.flags["images"]:
            return self.images_data
        else:
            raise Error("images is not set for this Dataset object")

    def return_fc(self):
        '''
        Returns fully-connected data.
        Should return error if flag indicates no fully-connected
        '''
        if self.flags["fc"]:
            return self.fc_data
        else:
            raise Error("fc is not set for this Dataset object")
        
