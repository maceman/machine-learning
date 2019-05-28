__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# CLASSIFICATION
class EcoliProcessing:

    """
    This class carries out the necessary instructions for processing the ecoli dataset.
    """

    def __init__(self, ecoli_path):
        self.ecoli_path = ecoli_path
        print("[ INFO ]: EcoliProcessing object created!")

    def preprocess_ecoli(self):

        """
        Preprocess the raw ecoli data.
        """

        print('[ INFO ]: Preprocessing ecoli data...')

        df = pd.read_csv(self.ecoli_path, header=None)
        df.columns = [
            'sequence_name','mcg','gvh','lip','chg','aac','alm1','alm2',
            'localization_site'
        ]
        quantitative_features = [
            'mcg','gvh','lip','chg','aac','alm1','alm2'
        ]
        categorical_features = [
            'sequence_name'
        ]
        class_var = 'localization_site'

        # Remove these observations from the dataset since they are sparse
        for cl in ['omL','imS','imL']:
            df = df[df[class_var] != cl]

        # Place classes into list
        classes = df[class_var].unique().tolist()
        df = alg.one_hot_encode(self, df, categorical_features)

        # Rearrange columns
        df = df[[x for x in df.columns if x != class_var] + [class_var]]

        return df, class_var, classes


    def ecoli_runner(self):

        """
        Execute the program runner over the ecoli dataset.
        """

        print('[ INFO ]: Initializing the ecoli program runner...')

        df, class_var, classes = self.preprocess_ecoli()
        x = alg()
        folds_dict = x.cross_validation(df, class_var, type='classification', folds=5)

        # Initialize comparion values
        best_fold_labels = None
        best_fold_score = 0
        best_fold_value = -1
        score = 0

        best_cond_fold_labels = None
        best_cond_fold_score = 0
        best_cond_fold_value = -1
        cond_score = 0

        # Loop through each fold in the folds dictionary
        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            best_k_labels = None
            best_k_score = 0
            best_k_value = -1

            for k in range(1, 16, 2):

                labels, score = x.knn(train_set, test_set, class_var, type='classification', k=k)

                # Determine if values need updating based on new results
                if score > best_k_score:

                    best_k_labels = labels
                    best_k_score = score
                    best_k_value = k

            # Determine if overall fold values need updating based on new results
            if best_k_score > best_fold_score:

                best_fold_labels = best_k_labels
                best_fold_score = best_k_score
                best_fold_value = best_k_value

        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            best_cond_k_labels = None
            best_cond_k_score = 0
            best_cond_k_value = -1

            for k in range(1, 16, 2):

                cond_train_set = x.condensed_knn(train_set, class_var)
                cond_labels, cond_score = x.knn(cond_train_set, test_set, class_var, type='classification', k=k)

                # Determine if values need updating based on new results for condensed version
                if cond_score > best_cond_k_score:

                    best_cond_k_labels = cond_labels
                    best_cond_k_score = cond_score
                    best_cond_k_value = k

            # Determine if overall fold values need updating based on new results for condensed version
            if best_cond_k_score > best_cond_fold_score:

                best_cond_fold_labels = best_cond_k_labels
                best_cond_fold_score = best_cond_k_score
                best_cond_fold_value = best_cond_k_value

        return best_fold_labels, best_fold_score, best_fold_value, best_cond_fold_labels, best_cond_fold_score, best_cond_fold_value
