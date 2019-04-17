'''
This will be a class that handles the preprocessing of data.
This class will preprocess data
'''
from .dataset import Dataset

import yaml

class Preprocessor:

    def __init__(self, flags, config_file="", drop_cols=["timestamp of pic", "timestamp when scrapped", "name", 'hastags/text']):
        '''
        Will initialize preprocessor

        Arguments:
        - flags: dict with bool values for all flags
                 these correspond to what of the data will we
                 use
                 Missing flag might raise error
        Potential flags:
        - "images": True/False
        - "fc": True/False

        - "normalize": True/False
        - "nlp_feats": True/False
        - "downsize": True/False
        '''
        if not (flags["images"] or flags["fc"]):
            print("Error: need to set at least one of images or fc")
            return -1
        self.flags = flags

        with open(config_file, 'r') as file:
            PARAMS = yaml.load(file)

        self.normalize = PARAMS["normalize"]
        self.nlp_feats = PARAMS["nlp_feats"]
        self.downsize = PARAMS["downsize"]
        self.drop_cols = drop_cols


    def fit(self, df):
        '''
        Will learn preprocessing based off of this dataframe

        Returns the preprocessed data as a Dataset instance

        Arguments:
        - filepath: filepath for the csv file that will have all names of all images
                    along with their features and stuff
        '''
        dataset = Dataset(self.flags)
        if self.flags["images"]:

            # Process images

            if self.downsize:
                # Make images smaller
                pass

            if self.normalize:
                # normalize images
                pass

        if self.flags["fc"]:

            # process features

            if self.nlp_feats:
                # Make NLP features
                pass

            if self.normalize:
                # normalize stuff
                pass

            fc_data = df.drop(self.drop_cols, axis=1)
            # fc_data = df[["followers"]]

            dataset.set_fc_data(fc_data)


        return dataset



    def transform(self, df):
        '''
        Will preprocess the data in the dataframe

        Returns preprocessed data as a Dataset instance

        Arguments:
        - filepath: filepath for the csv file that will have all names of all images
                    along with their features and stuff     
        '''

        dataset = Dataset(self.flags)
        if self.flags["images"]:

            # Process images

            if self.downsize:
                # Make images smaller
                pass

            if self.normalize:
                # normalize images
                pass

        if self.flags["fc"]:

            # process features

            if self.nlp_feats:
                # Make NLP features
                pass

            if self.normalize:
                # normalize stuff
                pass

            fc_data = df.drop(self.drop_cols, axis=1)
            # fc_data = df[["followers"]]

            dataset.set_fc_data(fc_data)


        return dataset
