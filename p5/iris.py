__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class IrisProcessing:

    """

    This class implements a procedure for training and testing three learning algorithms
    (Naive Bayes, Logistic Regression and Adaline) on the Iris Plants data set.

    """

    def __init__(self, iris_path):
        self.iris_path = iris_path
        print("[ INFO ]: IrisProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw iris data.
        """

        print('[ INFO ]: Preprocessing iris data...')

        # Rename headers of data frame
        iris_data = pd.read_csv(self.iris_path, header=None)
        iris_data.columns = [
            'sepal_length','sepal_width','petal_length','petal_width','iris_class'
        ]

        categorical_features = None
        continuous_features = [
            'sepal_length','sepal_width','petal_length','petal_width'
        ]
        predictor = 'iris_class'

        df = alg.continuous_to_discrete(self, iris_data, continuous_features)

        # Place classes into list
        classes = df[predictor].unique().tolist()

        features = [df.columns[f] for f in range(len(df.columns)) if df.columns[f] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the iris dataset.
        """

        print('[ INFO ]: Initializing the iris program runner...')

        df, features, predictor, classes = self.preprocess()

        x = alg()
        folds_dict = x.cross_validation(df, predictor, type='classification', folds=5)
        naive_bayes_results, naive_bayes_scores, naive_bayes_accuracy = x.runner_naive_bayes(predictor, classes, folds_dict)
        logistic_regression_results, logistic_regression_scores, logistic_regression_accuracy = x.runner_logistic_regression(predictor, classes, folds_dict)
        adaline_results, adaline_scores, adaline_accuracy = x.runner_adaline(predictor, classes, folds_dict)

        return naive_bayes_results, naive_bayes_scores, naive_bayes_accuracy, \
        logistic_regression_results, logistic_regression_scores, logistic_regression_accuracy, \
        adaline_results, adaline_scores, adaline_accuracy
