'''
This will be a class that handles the preprocessing of data.
This class will preprocess data
Note: This will only handle features, assumes labels are already removed from df
'''
from .dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image


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
            self.max_width = 0
            self.max_height = 0
            X = []
            for filename in df["name"]:
                img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
                if img.size[2] > self.max_height:
                    self.max_height = img.size[2]
                if img.size[1] > self.max_width:
                    self.max_width = img.size[1]               
                X.append(img)

            # Make all images the same size
            for idx, img in enumerate(X):
                img = img = img.resize((self.max_width, self.max_height), Image.ANTIALIAS)
                X[idx] = img_to_array(img) # this is a Numpy array


            if self.downsize:
                # Make images smaller
                pass

            X = np.asarray(X)

            if self.normalize:
                # normalize images
                self.img_mean = np.mean(X, axis=(1,2), keepdims=True)
                self.img_std = np.std(X, axis=(1,2), keepdims=True)
                X = (X - self.img_mean) / self.img_std

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
            X = []
            for filename in df["name"]:
                img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
                img = img.resize((self.max_width, self.max_height), Image.ANTIALIAS)
                x = img_to_array(img)  # this is a Numpy array
                X.append(x)


            if self.downsize:
                # Make images smaller
                pass

            X = np.asarray(X)

            if self.normalize:
                # normalize images
                X = (X - self.img_mean) / self.img_std

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
