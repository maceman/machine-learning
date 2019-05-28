__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# REGRESSION
class HardwareProcessing:

    """
    This class carries out the necessary instructions for processing the hardware dataset.
    """

    def __init__(self, hardware_path):
        self.hardware_path = hardware_path
        print("[ INFO ]: HardwareProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw hardware data.
        """

        print('[ INFO ]: Preprocessing hardware data...')

        # Rename headers of data frame
        hardware_data = pd.read_csv(self.hardware_path, header=None)
        hardware_data.columns = [
            ''
        ]
        categorical_features = [
            ''
        ]
        predictor = ''

        df = alg.one_hot_encode(self, hardware_data, categorical_features)

        classes = df[predictor].unique().tolist()

        features = [df.columns[j] for j in range(len(df.columns)) if df.columns[j] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the hardware dataset.
        """

        print('[ INFO ]: Initializing the hardware program runner...')

        df, features, predictor, classes = self.preprocess()
