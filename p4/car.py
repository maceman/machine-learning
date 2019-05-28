__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

# CLASSIFICATION
class CarProcessing:

    """
    This class carries out the necessary instructions for processing the car dataset.
    """

    def __init__(self, car_path):
        self.car_path = car_path
        print("[ INFO ]: CarProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw car data.
        """

        print('[ INFO ]: Preprocessing car data...')

        # Rename headers of data frame
        car_data = pd.read_csv(self.car_path, header=None)
        car_data.columns = [
            'buying','maint','doors','persons','lug_boot','safety','evaluation'
        ]
        categorical_features = [
            'buying','maint','doors','persons','lug_boot','safety'
        ]
        predictor = 'evaluation'

        df = alg.one_hot_encode(self, car_data, categorical_features)

        classes = car_data[predictor].unique().tolist()

        features = [df.columns[j] for j in range(len(df.columns)) if df.columns[j] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the abalone dataset.
        """

        print('[ INFO ]: Initializing the car program runner...')

        df, features, predictor, classes = self.preprocess()

        df = alg.random_feature_sample(self, df, 1.00)

        # Set up the training, testing and validation sets
        split = round(len(df) * 0.10)
        v_set = df[df.index < split]
        t_set = df[df.index >= split]

        tree = alg()
        folds_dict = tree.cross_validation(t_set, predictor, type='classification', folds=5)

        # Initialize comparion values
        best_fold_tree = None
        best_fold_score = 0
        best_fold_pred_labels = None
        best_fold_df = None

        # Loop through each fold in the folds dictionary
        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            # Build an ID3 tree
            root = tree.build_tree(train_set, features, predictor)
            df, labels, pred_labels, score = tree.test(test_set, features, predictor, root)

            # Determine which tree is the best
            if score > best_fold_score:
                best_fold_tree = root
                best_fold_score = score
                best_fold_pred_labels = pred_labels
                best_fold_df = df

        # Validate results and prune the ID3 tree
        v_tree = alg()
        df, labels, pred_labels, score = v_tree.test(v_set, features, predictor, best_fold_tree)
        prune_root = v_tree.prune(df, predictor, best_fold_tree)
        prune_df, prune_labels, prune_pred_labels, prune_score = v_tree.test(v_set, features, predictor, prune_root)

        return best_fold_tree, score, labels, pred_labels, prune_root, prune_score, prune_labels, prune_pred_labels
