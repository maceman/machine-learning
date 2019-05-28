__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# CLASSIFICATION
class SegmentationProcessing:

    """
    This class carries out the necessary instructions for processing the segmentation dataset.
    """

    def __init__(self, segmentation_path):
        self.segmentation_path = segmentation_path
        print("[ INFO ]: SegmentationProcessing object created!")

    def preprocess_segmentation(self):

        """
        Preprocess the raw segmentation data.
        """

        print('[ INFO ]: Preprocessing segmentation data...')

        # Rename headers of data frame
        segmentation_data = pd.read_csv(self.segmentation_path, header=0)
        quantitative_features = [segmentation_data.columns[j] for j in range(len(segmentation_data.columns)) if segmentation_data.columns[j] != 'IMAGE_CLASS']
        categorical_features = []
        class_var = 'IMAGE_CLASS'

        # Place classes into list
        classes = segmentation_data['IMAGE_CLASS'].unique().tolist()

        return segmentation_data, class_var, classes

    def segmentation_runner(self):

        """
        Execute the program runner over the segmentation dataset.
        """

        print('[ INFO ]: Initializing the segmentation program runner...')

        df, class_var, classes = self.preprocess_segmentation()
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
