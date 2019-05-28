w__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# REGRESSION
class WineProcessing:

    """
    This class carries out the necessary instructions for processing the wine dataset.
    """

    def __init__(self, wine_path):
        self.wine_path = wine_path
        print("[ INFO ]: WineProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw wine data.
        """

        print('[ INFO ]: Preprocessing wine data...')

        # Rename headers of data frame
        df = pd.read_csv(self.wine_path, header=0)
        df.columns = [
            'quality','alcohol','malic_acid','ash','alcalinity_acid','magnesium',
            'total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins',
            'color_intensity','hue','od280_od315','proline'
        ]
        predictor = 'quality'

        features = [df.columns[j] for j in range(len(df.columns)) if df.columns[j] != predictor]

        return df, features, predictor

    def runner(self):

        """
        Execute the program runner over the wine dataset.
        """

        print('[ INFO ]: Initializing the wine program runner...')

        df, features, predictor = self.preprocess()
