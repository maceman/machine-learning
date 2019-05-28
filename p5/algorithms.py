__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import cmath

class Algorithms:

    """
    This class serves as a holding ground for the algorithms needed to carry out
    the computations needed to score the cancer, iris, glass, soybean and vote
    datasets.

    Algorithms:

        -train_adaline
        -test_adaline
        -train_logistic_regression
        -test_logistic_regression
        -train_naive_bayes
        -test_naive_bayes
        -one_hot_encode_features
        -one_hot_encode_classes
        -continuous_to_discrete
        -randomize
        -cross_validation
        -random_feature_sample
        -runner_naive_bayes
        -runner_logistic_regression
        -runner_adaline

    """

    def __init__(self):
        print("[ INFO ]: Algorithm object created!")

    def train_adaline(self, train_set, classes, learning_rate):

        """
        Train the Adaline algorithm over the training set
        """

        print('[ INFO ]: Training with Adaline Classifier...')

        class_weights = {}

        for cl in classes:

            labels = train_set[cl]
            features = [x for x in train_set.columns if x not in classes]
            df = train_set[features]

            #initialize matrix operation
            np.random.seed(123)
            W = np.random.rand(1, len(features) + 1)
            leading_ones = np.ones((len(df), 1))
            mat = np.array(df)
            X = np.concatenate((mat, leading_ones), axis=1)

            # Loop over each training example to learn the optimzed weights
            for xi in range(len(X)):
                z = np.sum(np.multiply(W, X[xi]))
                W = W + (learning_rate * (labels[xi] - z) * X[xi])

            class_weights[cl] = W

        return class_weights

    def test_adaline(self, test_set, classes, class_weights):

        """
        Testing the Adaline algorithm over the testing set
        """

        print('[ INFO ]: Testing with Adaline Classifier...')

        true_samples_total = 0
        n_samples = 0

        scores = {}
        class_results = {}

        for cl, W in class_weights.items():

            b0 = W[0][-1]
            coeffs = W[0][:-1]
            labels = test_set[cl].tolist()
            features = [x for x in test_set.columns if x not in classes]

            # Apply the weights to the testing samples
            X = np.array(test_set[features])
            z = np.add(np.dot(X, coeffs), b0)

            # Compare against the threshold
            pred = z > 0.0
            pred = pred.astype(int).tolist()
            results = pd.DataFrame({'pred_class' : pred, 'true_class' : labels})

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            true_samples_total = true_samples_total + true_positives + true_negatives
            n_samples = n_samples + len(results)

            scores[cl] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives,
                'Classification Accuracy' : (true_positives + true_negatives) / len(results)
            }

            class_results[cl] = results

        classification_accuracy = true_samples_total / n_samples

        return class_results, scores, classification_accuracy

    def train_logistic_regression(self, train_set, classes):

        """
        Train the Logistic Regression algorithm over the training set
        """

        print('[ INFO ]: Training with Logistic Regression Classifier...')

        class_weights = {}

        for cl in classes:

            r = train_set[cl]
            features = [x for x in train_set.columns if x not in classes]
            df = train_set[features]

            #initialize matrix operations
            leading_ones = np.ones((len(df), 1))
            mat = np.array(df)
            X = np.concatenate((leading_ones, mat), axis=1)

            # Compute the optimization process for the weights
            XX_T = np.dot(X.T, X)
            XX_T_inv = np.linalg.pinv(XX_T)
            XX_T_inv_X_T = np.dot(XX_T_inv, X.T)
            W = np.dot(XX_T_inv_X_T, r)

            class_weights[cl] = W

        return class_weights

    def test_logistic_regression(self, test_set, classes, class_weights):

        """
        Testing the Logistic Regression algorithm over the testing set
        """

        print('[ INFO ]: Testing with Logistic Regression Classifier...')

        true_samples_total = 0
        n_samples = 0

        scores = {}
        class_results = {}

        for cl, W in class_weights.items():

            b0 = W[0]
            coeffs = W[1:]
            labels = test_set[cl].tolist()
            features = [x for x in test_set.columns if x not in classes]

            # Apply the weights to the testing samples
            X = np.array(test_set[features])
            z = np.add(np.dot(X, coeffs), b0)
            s = 1 / ( 1 + np.exp(-z))

            # Compare against the threshold
            pred = s > 0.4999
            pred = pred.astype(int).tolist()
            results = pd.DataFrame({'pred_class' : pred, 'true_class' : labels})

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            true_samples_total = true_samples_total + true_positives + true_negatives
            n_samples = n_samples + len(results)

            scores[cl] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives
            }

            class_results[cl] = results

        classification_accuracy = true_samples_total / n_samples

        return class_results, scores, classification_accuracy

    def train_naive_bayes(self, train_set, classes):

        """
        Train the Naive Bayes algorithm over the training set
        """

        print('[ INFO ]: Training with Naive Bayes Classifier...')

        class_probabilities = {}
        class_feature_probs = {}

        for cl in classes:

            feature_true_probs = {}
            feature_false_probs = {}

            # Find the probability that each class is in the training set
            class_probabilities[cl] = len(train_set[(train_set[cl] == 1)]) / len(train_set)

            # Compute the conditional feature probabilities based on the class probabilities
            # where the class is present
            class_true = train_set[(train_set[cl] == 1)]
            for col in class_true.columns:
                if col not in classes:
                    try:
                        true_true = len(class_true[(class_true[col] == 1)]) / len(class_true)
                    except:
                        true_true = 0
                    feature_true_probs[col] = true_true

            # Compute the conditional feature probabilities based on the class probabilities
            # where the class is not present
            class_false = train_set[(train_set[cl] == 0)]
            for col in class_false.columns:
                if col not in classes:
                    try:
                        false_false = len(class_false[(class_false[col] == 0)]) / len(class_false)
                    except:
                        false_false = 0
                    feature_false_probs[col] = false_false

            class_feature_probs[cl] = [feature_true_probs, feature_false_probs]

        return class_probabilities, class_feature_probs

    def test_naive_bayes(self, test_set, classes, class_probabilities, class_feature_probs):

        """
        Test the Naive Bayes classifier over the testing set
        """

        print('[ INFO ]: Testing with Naive Bayes Classifier...')

        class_results = {}
        scores = {}
        true_samples_total = 0
        n_samples = 0

        for cl in classes:

            # Create new column for class predictions
            feature_set = test_set.drop(classes, axis=1)
            feature_set['pred_class'] = 0
            true_class = test_set[cl]

            for row in range(len(feature_set)):

                # Initialize probability sums for each class
                true_probs_sum = 1
                false_probs_sum = 1
                true_conditional_prob_sum = 1
                false_conditional_prob_sum = 1

                for col in feature_set.columns:

                    if col != 'pred_class':

                        # Calculate probabilities assuming the class is present or 1
                        if feature_set[col].iloc[row] == 1:

                            # Compute conditional feature probabilities based on
                            # wether or not the feature is present (1 or 0)
                            true_prob = class_feature_probs[cl][0].get(col)
                            false_prob = 1 - class_feature_probs[cl][1].get(col)

                        else:

                            # Calculate probabilities assuming the class is not present or 0
                            true_prob = 1 - class_feature_probs[cl][0].get(col)
                            false_prob = class_feature_probs[cl][1].get(col)

                        # Multiply all feature probabilities together for each record
                        true_probs_sum = true_probs_sum * true_prob
                        false_probs_sum = false_probs_sum * false_prob

                # Multiply class conditional probabilities by conditional feature probabilities
                true_conditional_prob_sum = class_probabilities[cl] * true_probs_sum
                false_conditional_prob_sum = (1 - class_probabilities[cl]) * false_probs_sum

                # Determine which probability is highest - highest one is selected as the prediction value
                if true_conditional_prob_sum > false_conditional_prob_sum:
                    feature_set['pred_class'].iloc[row] = 1

            # Place the results into a data frame for comparison
            results = pd.concat([feature_set['pred_class'], true_class], axis=1)
            results.columns = ['pred_class', 'true_class']
            class_results[cl] = results

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            scores[cl] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives
            }

            true_samples_total = true_samples_total + true_positives + true_negatives
            n_samples = n_samples + len(results)

        classification_accuracy = true_samples_total / n_samples

        return class_results, scores, classification_accuracy

    def one_hot_encode_features(self, df, categorical_data):

        """
        One-hot-enode all categorical features
        """

        print('[ INFO ]: Conducting One-Hot-Encoding...')

        for col in df.columns:
            if col in categorical_data:
                for value in df[col].unique():
                    df[str(col) + '_' + str(value)] = np.where(df[col] == value, 1, 0)
                df = df.drop(col, axis=1)

        return df

    def one_hot_encode_classes(self, df, predictor, classes):

        # Create new columns for each unique class
        new_column_classes = []
        for cl in classes:

            df['{}_class'.format(cl)] = np.where(df[predictor] == cl, 1, 0)
            #classes = np.char.replace(classes, cl, str(cl) + '_class')
            new_column_classes.append('{}_class'.format(cl))
        df = df.drop(predictor, axis=1)
        #print(classes)
        #print(new_column_classes)
        return df, new_column_classes

    def continuous_to_discrete(self, df, continuous_features):

        """
        Convert continuous features to discrete features in the raw data set
        """

        print('[ INFO ]: Performing Continuous-to-discrete on the raw data...')

        for col in continuous_features:

                # Find mean and std dev of each column
                col_mean = np.mean(df[col])
                col_std  = np.std(df[col])

                # Build out column distribution based on mean and std dev values
                col_dist = [
                    col_mean - (2 * col_std),
                    col_mean - col_std,
                    col_mean,
                    col_mean + col_std,
                    col_mean + (2 * col_std)
                ]

                # Create new columns to support binary features
                df[col + '_less_n2std'] = np.where(df[col] < col_dist[0], 1, 0)
                df[col + '_btw_n2std_n1std'] = np.where(np.logical_and(df[col] < col_dist[1], df[col] >= col_dist[0]), 1, 0)
                df[col + '_btw_n1std_mean'] = np.where(np.logical_and(df[col] < col_dist[2], df[col] >= col_dist[1]), 1, 0)
                df[col + '_btw_mean_p1std'] = np.where(np.logical_and(df[col] < col_dist[3], df[col] >= col_dist[2]), 1, 0)
                df[col + '_btw_p1std_p2std'] = np.where(np.logical_and(df[col] < col_dist[4], df[col] >= col_dist[3]), 1, 0)
                df[col + '_more_p2std'] = np.where(df[col] >= col_dist[4], 1, 0)
                df = df.drop(col, axis=1)

        return df

    def randomize(self, df):

        """
        Randomize the observations in the raw data
        """

        print('[ INFO ]: Randomizing data...')

        np.random.seed(123)
        df['rand'] = np.random.rand(len(df))
        df = df.sort_values(by=['rand']).reset_index()
        df = df.drop(['rand','index'], axis=1)

        print('[ INFO ]: Finished randomizing data...')

        return df

    def cross_validation(self, df, class_var, type, folds):

        """
        Carry out cross-validation for a given training set
        """

        print('[ INFO ]: Running {}-fold cross-validation...'.format(folds))

        folds_dict = {}
        for fold in range(folds):
            folds_dict['fold_' + str(fold + 1)] = pd.DataFrame()

        # If prediction type is regression, randomly split dataset into n folds
        if type == 'regression':

            df = self.randomize(df)
            fold_len = round(len(df) / folds)

            for fold in range(folds):

                x = df[df.index <= fold_len]
                folds_dict['fold_' + str(fold + 1)] = x
                df = df[~df.isin(x)].dropna()
                df = df.reset_index()
                df = df.drop(['index'], axis=1)

        # Otherwise, create folds based on likelihood of each class in each fold
        else:

            s = round(df[class_var].value_counts() / folds)
            df = self.randomize(df)

            for cl, v in s.items():

                x = df[df[class_var] == cl]

                for fold in range(folds):

                    x = x.reset_index()
                    y = x[x.index <= v]

                    # Add fold to folds dictionary for processing
                    folds_dict['fold_' + str(fold + 1)] = folds_dict['fold_' + str(fold + 1)].append(y, ignore_index=True)

                    x = x[~x.isin(y)].dropna()
                    x = x.reset_index()
                    x = x.drop(['index','level_0'], axis=1)

            # Remove final index column
            for fold in folds_dict:
                folds_dict[fold] = folds_dict[fold].drop(['index'], axis=1)

        print('[ INFO ]: Finished {}-fold cross-validation...'.format(folds))

        return folds_dict

    def random_feature_sample(self, df, frac):

        """
        Produce a random sample of the provided dataset for enhanced clustering computations
        """

        print('[ INFO ]: Normalizing data...')

        random_sample_data = df.sample(frac=frac, random_state=1)

        return random_sample_data

    def runner_naive_bayes(self, predictor, classes, folds_dict):

        """
        Run the procedures for the naive bayes algorithm.
        """

        print('[ INFO ]: Initiate Naive Bayes runner...')

        best_fold_class_results = None
        best_fold_scores = None
        best_classification_accuracy = 0

        # Loop through each fold in the folds dictionary
        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            train_set, new_class_columns = self.one_hot_encode_classes(train_set, predictor, classes)
            test_set, new_class_columns = self.one_hot_encode_classes(test_set, predictor, classes)

            # Carry out the procedures for naive bayes
            class_probabilities, class_feature_probs = self.train_naive_bayes(train_set, new_class_columns)
            class_results, scores, classification_accuracy = self.test_naive_bayes(test_set, new_class_columns, class_probabilities, class_feature_probs)

            # Determine which tree is the best
            if classification_accuracy > best_classification_accuracy:
                best_fold_class_results = class_results
                best_fold_scores = scores
                best_classification_accuracy = classification_accuracy

        return best_fold_class_results, best_fold_scores, best_classification_accuracy

    def runner_logistic_regression(self, predictor, classes, folds_dict):

        """
        Run the procedures for the logistic regression algorithm.
        """

        print('[ INFO ]: Initiate Logistic Regression runner...')

        best_fold_class_results = None
        best_fold_scores = None
        best_classification_accuracy = 0

        # Loop through each fold in the folds dictionary
        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            train_set, new_class_columns = self.one_hot_encode_classes(train_set, predictor, classes)
            test_set, new_class_columns = self.one_hot_encode_classes(test_set, predictor, classes)

            # Carry out the procedures for logistic regression
            class_weights = self.train_logistic_regression(train_set, new_class_columns)
            class_results, scores, classification_accuracy = self.test_logistic_regression(test_set, new_class_columns, class_weights)

            # Determine which tree is the best
            if classification_accuracy > best_classification_accuracy:
                best_fold_class_results = class_results
                best_fold_scores = scores
                best_classification_accuracy = classification_accuracy

        return best_fold_class_results, best_fold_scores, best_classification_accuracy

    def runner_adaline(self, predictor, classes, folds_dict):

        """
        Run the procedures for the adaline algorithm.
        """

        print('[ INFO ]: Initiate Adaline runner...')

        best_fold_class_results = None
        best_fold_scores = None
        best_classification_accuracy = 0

        # Loop through each fold in the folds dictionary
        for fold in folds_dict:

            test_set = folds_dict[fold]
            train_set = pd.DataFrame()
            for inner_fold in folds_dict:
                if inner_fold != fold:
                    train_set = train_set.append(folds_dict[inner_fold], ignore_index=True)

            train_set, new_class_columns = self.one_hot_encode_classes(train_set, predictor, classes)
            test_set, new_class_columns = self.one_hot_encode_classes(test_set, predictor, classes)

            # Carry out the procedures for adaline
            class_weights = self.train_adaline(train_set, new_class_columns, 0.01)
            class_results, scores, classification_accuracy = self.test_adaline(test_set, new_class_columns, class_weights)

            # Determine which tree is the best
            if classification_accuracy > best_classification_accuracy:
                best_fold_class_results = class_results
                best_fold_scores = scores
                best_classification_accuracy = classification_accuracy

        return best_fold_class_results, best_fold_scores, best_classification_accuracy
