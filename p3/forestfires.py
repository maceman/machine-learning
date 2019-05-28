__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# REGRESSION
class ForestfiresProcessing:

    """
    This class carries out the necessary instructions for processing the forestfires dataset.
    """

    def __init__(self, forestfires_path):
        self.forestfires_path = forestfires_path
        print("[ INFO ]: ForestfiresProcessing object created!")

    def preprocess_forestfires(self):

        """
        Preprocess the raw forestfires data.
        """

        print('[ INFO ]: Preprocessing forestfires data...')

        # Rename headers of data frame
        forestfires_data = pd.read_csv(self.forestfires_path, header=0)
        forestfires_data.columns = [
            'x_axis','y_axis','month','day','ffmc','dmc','dc','isi','temp','rh',
            'wind','rain','area'
        ]
        quantitative_features = [
            'x_axis','y_axis','ffmc','dmc','dc','isi','temp','rh','wind','rain','area'
        ]
        categorical_features = [
            'month','day'
        ]
        response_var = 'area'

        df = alg.one_hot_encode(self, forestfires_data, categorical_features)

        return df, response_var

    def forestfires_runner(self):

        """
        Execute the program runner over the forestfires dataset.
        """

        print('[ INFO ]: Initializing the forestfires program runner...')

        df, response_var = self.preprocess_forestfires()
        x = alg()
        folds_dict = x.cross_validation(df, response_var, type='regression', folds=5)

        # Initialize comparison values
        best_fold_labels = None
        best_fold_score = 1000000
        best_fold_value = -1

        # Loop through each fold in the folds dictionary
        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            best_k_labels = None
            best_k_score = 1000000
            best_k_value = -1

            for k in range(1, 16, 2):

                labels, score = x.knn(train_set, test_set, response_var, type='regression', k=k)

                # Determine if values need updating based on new results
                if score < best_k_score:

                    best_k_labels = labels
                    best_k_score = score
                    best_k_value = k

            # Determine if overall fold values need updating based on new results
            if best_k_score < best_fold_score:

                best_fold_labels = best_k_labels
                best_fold_score = best_k_score
                best_fold_value = best_k_value

        return best_fold_labels, best_fold_score, best_fold_value
