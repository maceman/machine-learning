__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class GlassProcessing:

    """

    This class implements a procedure for training and testing three learning algorithms
    (Naive Bayes, Logistic Regression and Adaline) on the Glass Identification data set.

    """

    def __init__(self, glass_path):
        self.glass_path = glass_path
        print("[ INFO ]: GlassProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw glass data.
        """

        print('[ INFO ]: Preprocessing glass data...')

        # Rename headers of data frame
        glass_data = pd.read_csv(self.glass_path, header=None)
        glass_data.columns = [
            'id_number','refractive_index','sodium','magnesium',
            'aluminum','silicon','potassium',
            'calcium','barium','iron','class'
        ]

        categorical_features = None
        continuous_features = [
            'refractive_index','sodium','magnesium',
            'aluminum','silicon','potassium',
            'calcium','barium','iron'
        ]
        predictor = 'class'

        df = alg.continuous_to_discrete(self, glass_data, continuous_features)

        df = df.drop(['id_number'], axis=1)

        # Place classes into list
        classes = df[predictor].unique().tolist()

        features = [df.columns[f] for f in range(len(df.columns)) if df.columns[f] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the glass dataset.
        """

        print('[ INFO ]: Initializing the glass program runner...')

        df, features, predictor, classes = self.preprocess()

        x = alg()
        folds_dict = x.cross_validation(df, predictor, type='classification', folds=5)
        naive_bayes_results, naive_bayes_scores, naive_bayes_accuracy = x.runner_naive_bayes(predictor, classes, folds_dict)
        logistic_regression_results, logistic_regression_scores, logistic_regression_accuracy = x.runner_logistic_regression(predictor, classes, folds_dict)
        adaline_results, adaline_scores, adaline_accuracy = x.runner_adaline(predictor, classes, folds_dict)

        return naive_bayes_results, naive_bayes_scores, naive_bayes_accuracy, \
        logistic_regression_results, logistic_regression_scores, logistic_regression_accuracy, \
        adaline_results, adaline_scores, adaline_accuracy
