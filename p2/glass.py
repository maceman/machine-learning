__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class GlassProcessing:

    """
    This class carries out the necessary instructions for processing the glass dataset.
    """

    def __init__(self, glass_path):
        self.glass_path = glass_path
        print("GlassProcessing object created!")

    def preprocess_glass(self):

        """
        Preprocess the raw glass data.
        """

        print('[ INFO ]: Preprocessing glass data...')

        # Rename headers of data frame
        glass_data = pd.read_csv(self.glass_path, header=None)
        glass_data.columns = ['id_number','refractive_index','sodium','magnesium',
                     'aluminum','silicon','potassium',
                     'calcium','barium','iron','glass_class']
        glass_data = glass_data.drop(['id_number'], axis=1)

        df_columns = [glass_data.columns[j] for j in range(len(glass_data.columns)) if glass_data.columns[j] != 'glass_class']

        # Place classes into list
        classes = glass_data['glass_class'].unique().tolist()

        return glass_data, df_columns, classes

    def glass_runner(self):

        """
        Execute the program runner over the glass dataset.
        """

        print('[ INFO ]: Initializing the glass program runner...')

        data, features, classes = self.preprocess_glass()
        glass = alg()
        random_data = glass.random_feature_sample(data, 0.20)
        selected_features, selected_clusters, basePerformance = glass.stepwise_forward_selection(random_data, features, len(classes))
        glass_clusters, glass_performance = glass.recompute_stepwise_with_population(data, selected_features, len(classes))

        return selected_features, glass_clusters, basePerformance, glass_performance
