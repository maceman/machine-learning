__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# REGRESSION
class MachineProcessing:

    """
    This class carries out the necessary instructions for processing the machine dataset.
    """

    def __init__(self, machine_path):
        self.machine_path = machine_path
        print("[ INFO ]: MachineProcessing object created!")

    def preprocess_machine(self):

        """
        Preprocess the raw machine data.
        """

        print('[ INFO ]: Preprocessing machine data...')

        # Rename headers of data frame
        machine_data = pd.read_csv(self.machine_path, header=None)
        machine_data.columns = [
            'vendor','model_name','myct','mmin','mmax','cach','chmin','chmax',
            'prp','erp'
        ]
        quantitative_features = [
            'myct','mmin','mmax','cach','chmin','chmax','erp','prp'
        ]
        categorical_features = [
            'vendor','model_name'
        ]
        response_var = 'prp'

        df = alg.one_hot_encode(self, machine_data, categorical_features)

        return df, response_var

    def machine_runner(self):

        """
        Execute the program runner over the machine dataset.
        """

        print('[ INFO ]: Initializing the machine program runner...')

        df, response_var = self.preprocess_machine()
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
