'''
This will be a class that handles the preprocessing of data.
This class will preprocess data
Note: This will only handle features, assumes labels are already removed from df
'''
from .dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import numpy as np

import yaml
from zipfile import ZipFile
import io
import gc

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
            PARAMS = yaml.load(file, Loader=yaml.FullLoader)

        self.normalize = PARAMS["normalize"]
        self.nlp_feats = PARAMS["nlp_feats"]
        self.downsize = PARAMS["downsize"]
        self.drop_cols = drop_cols
        self.target_size = PARAMS["img_size"]


    def fit(self, df, zip_file="./all_data.zip"):
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
            X = []
            with ZipFile(zip_file) as archive:
                for filename in df["name"]:
                    imgdata = archive.read("all_data/" + filename)
                    img = Image.open(io.BytesIO(imgdata)).resize(self.target_size)
                    # img = Image.frombytes(mode="RGB", size=self.target_size, data=imgdata, decoder_name="raw")
                    # img = load_img(path + filename, target_size=self.target_size)  # this is a PIL image
                    x = img_to_array(img) # this is a Numpy array
                    x = np.reshape(x, (x.shape[2], x.shape[1], x.shape[0]))
                    # print(x.shape)          
                    X.append(x)

            if self.downsize:
                # Make images smaller
                pass

            gc.collect()
            X = np.asarray(X)

            if self.normalize:
                # normalize images
                self.img_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
                self.img_std = np.std(X, axis=(0, 2, 3), keepdims=True)
                X = (X - self.img_mean) / self.img_std

            print(X.shape)

            dataset.set_images(X)

        if self.flags["fc"]:

            # process features

            if self.nlp_feats:
                # Make NLP features
                pass

            fc_data = df.drop(self.drop_cols, axis=1)

            if self.normalize:
                # normalize stuff
                self.fc_mean = df.mean()
                self.fc_std = df.std()
                df = (self.fc_mean) / self.fc_std

            # fc_data = df[["followers"]]

            dataset.set_fc_data(fc_data)


        return dataset



    def transform(self, df, zip_file="./all_data.zip"):
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
            X = []
            with ZipFile(zip_file) as archive:
                for filename in df["name"]:
                    imgdata = archive.read("all_data/" + filename)
                    img = Image.open(io.BytesIO(imgdata)).resize(self.target_size)
                    # img = Image.frombytes(mode="RGB", size=self.target_size, data=imgdata, decoder_name="raw")
                    # img = load_img(path + filename, target_size=self.target_size)  # this is a PIL image
                    x = img_to_array(img) # this is a Numpy array
                    x = np.reshape(x, (x.shape[2], x.shape[1], x.shape[0]))
                    # print(x.shape)          
                    X.append(x)


            if self.downsize:
                # Make images smaller
                pass

            gc.collect()
            X = np.asarray(X)

            if self.normalize:
                # normalize images
                X = (X - self.img_mean) / self.img_std

            print(X.shape)

            dataset.set_images(X)

        if self.flags["fc"]:

            # process features

            if self.nlp_feats:
                # Make NLP features
                pass

            fc_data = df.drop(self.drop_cols, axis=1)

            if self.normalize:
                # normalize stuff
                df = (self.fc_mean) / self.fc_std

            # fc_data = df[["followers"]]

            dataset.set_fc_data(fc_data)


        return dataset
