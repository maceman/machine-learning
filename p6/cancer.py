__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class CancerProcessing():

    """

    This class implements a procedure for training and testing three learning algorithms
    (Naive Bayes, Logistic Regression and Adaline) on the Breast Cancer Wisconsin data set.

    """

    def __init__(self, cancer_path):
        self.cancer_path = cancer_path
        print("[ INFO ]: CancerProcessing object created!")

    def preprocess(self):

        """
        Preprocess the raw cancer data.
        """

        print('[ INFO ]: Preprocessing cancer data...')

        # Rename headers of data frame
        cancer_data = pd.read_csv(self.cancer_path, header=None)
        cancer_data.columns = [
            'sample_code','clump_thickness','unif_cell_size','unif_cell_shape',
            'marginal_adhesion','single_epi_cell_size','bare_nuclei',
            'bland_chromatin','normal_nucleoli','mitoses','class'
        ]
        cancer_data = cancer_data.drop('sample_code', axis=1)

        # Impute missing data
        bare_nuclei_mean = cancer_data['bare_nuclei'].loc[cancer_data['bare_nuclei'] != '?']
        bare_nuclei_mean = round(np.mean(bare_nuclei_mean.astype(int))).astype(int)
        cancer_data['bare_nuclei_imp'] = np.where(cancer_data['bare_nuclei'] == '?', str(bare_nuclei_mean), cancer_data['bare_nuclei']).astype(int)
        df = cancer_data.drop('bare_nuclei', axis=1)

        categorical_features = [
            'clump_thickness','unif_cell_size','unif_cell_shape',
            'marginal_adhesion','single_epi_cell_size','bare_nuclei_imp',
            'bland_chromatin','normal_nucleoli','mitoses'
        ]
        continuous_features = None
        predictor = 'class'

        df = alg.one_hot_encode_features(self, df, categorical_features)

        # Place classes into list
        classes = df[predictor].unique().tolist()

        features = [df.columns[f] for f in range(len(df.columns)) if df.columns[f] != predictor]

        return df, features, predictor, classes

    def runner(self):

        """
        Execute the program runner over the cancer dataset.
        """

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
