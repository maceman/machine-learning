__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# REGRESSION
class ForestfiresProcessing:

    """
    This class carries out the necessary instructions for processing the forest fires dataset.
    """

    def __init__(self, forestfires_path):
        self.forestfires_path = forestfires_path
        print("[ INFO ]: ForestfiresProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw forest fires data.
        """

        print('[ INFO ]: Preprocessing forest fires data...')

        # Rename headers of data frame
        forestfires_data = pd.read_csv(self.forestfires_path, header=0)
        forestfires_data.columns = [
            'x_axis','y_axis','month','day','ffmc','dmc','dc','isi','temp','rh',
            'wind','rain','area'
        ]
        categorical_features = [
            'month','day'
        ]
        predictor = 'area'

        df = alg.one_hot_encode(self, forestfires_data, categorical_features)

        features = [df.columns[j] for j in range(len(df.columns)) if df.columns[j] != predictor]

        return df, features, predictor

    def runner(self):

        """
        Execute the program runner over the forestfires dataset.
        """

        print('[ INFO ]: Initializing the forest fires program runner...')

        df, features, predictor = self.preprocess()
