__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class SoyProcessing:

    """

    This class implements a procedure for training and testing three learning algorithms
    (Naive Bayes, Logistic Regression and Adaline) on the Small Soybean data set.

    """

    def __init__(self, soybean_path):
        self.soybean_path = soybean_path
        print("[ INFO ]: SoyProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw soy data.
        """

        print('[ INFO ]: Preprocessing soy data...')

        # Rename headers of data frame
        soy_data = pd.read_csv(self.soybean_path, header=None)
        soy_data.columns = ['attr_{}'.format(i) for i in range(0,len(soy_data.columns))]


        categorical_features = [
            f for f in soy_data.columns if f != 'attr_35'
        ]
        continuous_features = None
        predictor = 'attr_35'

        df = alg.one_hot_encode_features(self, soy_data, categorical_features)

        # Place classes into list
        classes = df[predictor].unique().tolist()

        features = [df.columns[f] for f in range(len(df.columns)) if df.columns[f] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the soybean dataset.
        """

        print('[ INFO ]: Initializing the soybean program runner...')

        df, features, predictor, classes = self.preprocess()

        x = alg()
        folds_dict = x.cross_validation(df, predictor, type='classification', folds=5)
        no_fold_class_results, no_fold_scores, no_classification_accuracy, no_model = x.runner_nn_no_layer(predictor, classes, folds_dict)
        one_fold_class_results, one_fold_scores, one_classification_accuracy, one_model = x.runner_nn_one_layer(predictor, classes, folds_dict)
        two_fold_class_results, two_fold_scores, two_classification_accuracy, two_model = x.runner_nn_two_layer(predictor, classes, folds_dict)

        return no_fold_class_results, no_fold_scores, no_classification_accuracy, no_model, \
        one_fold_class_results, one_fold_scores, one_classification_accuracy, one_model, \
        two_fold_class_results, two_fold_scores, two_classification_accuracy, two_model
