__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class VoteProcessing:

    """

    This class implements a procedure for training and testing three learning algorithms
    (Naive Bayes, Logistic Regression and Adaline) on the 1984 United States Congressional
    Voting Record data set.

    """

    def __init__(self, vote_path):
        self.vote_path = vote_path
        print("[ INFO ]: VoteProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw vote data.
        """

        print('[ INFO ]: Preprocessing vote data...')

        # Rename headers of data frame
        vote_data = pd.read_csv(self.vote_path, header=None)
        vote_data.columns = ['class','handicapped_infant','water_project_cost_sharin','adoption_of_the_budget_resolution',
                    'physician_fee_freeze','el_salvador_aid','religious_groups_in_schools',
                    'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration',
                    'synfuels_corporation_cutback','education_spending','superfund_right_to-sue',
                    'crime','duty_free_exports','export_administration_act_south_africa']

        # Substitute '?' for 'm' which stands for 'maybe'
        vote_data = vote_data.replace('?', 'm')

        categorical_features = [
            f for f in vote_data.columns if f != 'class'
        ]
        continuous_features = None
        predictor = 'class'

        df = alg.one_hot_encode_features(self, vote_data, categorical_features)

        # Place classes into list
        classes = df[predictor].unique().tolist()

        features = [df.columns[f] for f in range(len(df.columns)) if df.columns[f] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the vote dataset.
        """

        print('[ INFO ]: Initializing the vote program runner...')

        print('[ INFO ]: Initializing the cancer program runner...')

        df, features, predictor, classes = self.preprocess()

        x = alg()
        folds_dict = x.cross_validation(df, predictor, type='classification', folds=5)
        no_fold_class_results, no_fold_scores, no_classification_accuracy, no_model = x.runner_nn_no_layer(predictor, classes, folds_dict)
        one_fold_class_results, one_fold_scores, one_classification_accuracy, one_model = x.runner_nn_one_layer(predictor, classes, folds_dict)
        two_fold_class_results, two_fold_scores, two_classification_accuracy, two_model = x.runner_nn_two_layer(predictor, classes, folds_dict)

        return no_fold_class_results, no_fold_scores, no_classification_accuracy, no_model, \
        one_fold_class_results, one_fold_scores, one_classification_accuracy, one_model, \
        two_fold_class_results, two_fold_scores, two_classification_accuracy, two_model
