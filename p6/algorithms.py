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
    the computations necessary to score the cancer, iris, glass, soybean and vote
    datasets using various architects of neural networks.

    Algorithms:

        -train_nn_two_layer
        -test_nn_two_layer
        -train_nn_one_layer
        -test_nn_one_layer
        -train_nn_no_layer
        -test_nn_no_layer
        -one_hot_encode_features
        -one_hot_encode_classes
        -continuous_to_discrete
        -randomize
        -cross_validation
        -random_feature_sample
        -forward_propagate
        -sigmoid
        -compute_cost
        -output_layer_error
        -hidden_layer_error
        -sigmoid_derivative
        -runner_nn_no_layer
        -runner_nn_one_layer
        -runner_nn_two_layer

    """

    def __init__(self):
        print("[ INFO ]: Algorithm object created!")

    def train_nn_two_layer(self, train_set, classes, learning_rate, n_nodes_array):

        class_weights = {}

        for cl in classes:

            Y = np.array(train_set[cl])
            features = [x for x in train_set.columns if x not in classes]
            df = train_set[features]

            # Initialize variables
            X = np.array(df)
            W0 = np.random.rand(len(X[0]), n_nodes_array[0])
            W1 = np.random.rand(n_nodes_array[0], n_nodes_array[1])
            W2 = np.random.rand(n_nodes_array[1], 1)

            b0 = np.zeros((1, n_nodes_array[0]))
            b1 = np.zeros((1, n_nodes_array[1]))
            b2 = np.zeros((1, 1))

            # Loop through all training examples
            for i in range(len(X)):

                # Compute first hidden layer of network
                z0 = self.forward_propagate(X[i], W0, b0)
                s0 = self.sigmoid(z0)

                # Compute second hidden layer of network
                z1 = self.forward_propagate(s0, W1, b1)
                s1 = self.sigmoid(z1)

                # Compute outout from single hidden layer
                z2 = self.forward_propagate(s1, W2, b2)
                s2 = np.squeeze(self.sigmoid(z2))

                output_error = self.output_layer_error(Y[i], s2)

                hidden_error_2 = self.hidden_layer_error(W1, output_error, s1)

                hidden_error_1 = self.hidden_layer_error(W0, output_error, s0)

                W0 = W0 + learning_rate * hidden_error_1 * X[i].reshape(len(X[i]), 1)
                W1 = W1 + learning_rate * hidden_error_2 * s0.reshape(n_nodes_array[0], 1)
                W2 = W2 + learning_rate * output_error * s1.reshape(n_nodes_array[1], 1)

            class_weights[cl] = {
                'W0': W0,
                'W1': W1,
                'W2': W2,
                'b0': b0,
                'b1': b1,
                'b2': b2
            }

        return class_weights

    def test_nn_two_layer(self, test_set, classes, class_weights, n_nodes_array):

        true_samples_total = 0
        n_samples = 0

        scores = {}
        class_results = {}

        for cl, weights in sorted(class_weights.items()):

            labels = test_set[cl].tolist()
            features = [x for x in test_set.columns if x not in classes]

            W0 = weights['W0']
            W1 = weights['W1']
            W2 = weights['W2']
            b0 = weights['b0']
            b1 = weights['b1']
            b2 = weights['b2']

            # Apply the weights to the testing samples
            X = np.array(test_set[features])

            pred = np.zeros((len(X), 1))

            for i in range(len(X)):

                z0 = self.forward_propagate(X[i], W0, b0)
                s0 = self.sigmoid(z0)

                z1 = self.forward_propagate(s0, W1, b1)
                s1 = self.sigmoid(z1)


                z2 = self.forward_propagate(s1, W2, b2)
                s2 = np.squeeze(self.sigmoid(z2))

                if s2 > 0.96:
                    pred[i] = 1
                else:
                    pred[i] = 0

            pred = np.squeeze(pred)
            results = pd.DataFrame({'pred_class' : pred, 'true_class' : labels})

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            true_samples_total = true_samples_total + true_positives + true_negatives
            n_samples = n_samples + len(results)
            #print(results)

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


    def train_nn_one_layer(self, train_set, classes, learning_rate, n_nodes):

        class_weights = {}

        for cl in classes:

            Y = np.array(train_set[cl])
            features = [x for x in train_set.columns if x not in classes]
            df = train_set[features]

            # Initialize variables
            X = np.array(df)
            W0 = np.random.rand(len(X[0]), n_nodes)
            W1 = np.random.rand(n_nodes, 1)
            b0 = np.zeros((1, n_nodes))
            b1 = np.zeros((1, 1))

            # Loop through all training examples
            for i in range(len(X)):

                # Compute first hidden layer of network
                z0 = self.forward_propagate(X[i], W0, b0)
                s0 = self.sigmoid(z0)

                # Compute outout from single hidden layer
                z1 = self.forward_propagate(s0, W1, b1)
                s1 = np.squeeze(self.sigmoid(z1))

                output_error = self.output_layer_error(Y[i], s1)

                hidden_error = self.hidden_layer_error(W0, output_error, s0)

                W0 = W0 + learning_rate * hidden_error * X[i].reshape(len(X[i]), 1)
                W1 = W1 + learning_rate * output_error * s0.reshape(n_nodes, 1)


            class_weights[cl] = {
                'W0': W0,
                'W1': W1,
                'b0': b0,
                'b1': b1
            }

        return class_weights

    def test_nn_one_layer(self, test_set, classes, class_weights, n_nodes):

        true_samples_total = 0
        n_samples = 0

        scores = {}
        class_results = {}

        for cl, weights in sorted(class_weights.items()):

            labels = test_set[cl].tolist()
            features = [x for x in test_set.columns if x not in classes]

            W0 = weights['W0']
            W1 = weights['W1']
            b0 = weights['b0']
            b1 = weights['b1']

            # Apply the weights to the testing samples
            X = np.array(test_set[features])
            pred = np.zeros((len(X), 1))

            for i in range(len(X)):

                z0 = self.forward_propagate(X[i], W0, b0)
                s0 = self.sigmoid(z0)

                z1 = self.forward_propagate(s0, W1, b1)
                s1 = np.squeeze(self.sigmoid(z1))
                #print('S1',s1)

                if s1 > 0.96:
                    pred[i] = 1
                else:
                    pred[i] = 0

            pred = np.squeeze(pred)
            results = pd.DataFrame({'pred_class' : pred, 'true_class' : labels})

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            true_samples_total = true_samples_total + true_positives + true_negatives
            n_samples = n_samples + len(results)
            #print(results)

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

    def train_nn_no_layer(self, train_set, classes, learning_rate):

        class_weights = {}

        for cl in classes:

            Y = np.array(train_set[cl])
            features = [x for x in train_set.columns if x not in classes]
            df = train_set[features]

            # Initialize variables
            X = np.array(df)
            W = np.squeeze(np.random.rand(1, len(X[0])))
            b = np.squeeze(np.random.rand(1,1))

            for i in range(len(X)):

                z = np.dot(X[i], W) + b

                s = self.sigmoid(z)

                W = W + learning_rate * (Y[i] - s) * X[i]
                b = b + learning_rate * (Y[i] - s)

            class_weights[cl] = {
                'W': W,
                'b': b
            }

        return class_weights

    def test_nn_no_layer(self, test_set, classes, class_weights):

        true_samples_total = 0
        n_samples = 0

        scores = {}
        class_results = {}

        for cl, weights in sorted(class_weights.items()):

            labels = test_set[cl].tolist()
            features = [x for x in test_set.columns if x not in classes]

            W = weights['W']
            b = weights['b']

            # Apply the weights to the testing samples
            X = np.array(test_set[features])
            z = np.add(np.dot(X, W), b)

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
            new_column_classes.append('{}_class'.format(cl))

        df = df.drop(predictor, axis=1)

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

            for cl, v in sorted(s.items()):

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

        random_sample_data = df.sample(frac=frac)

        return random_sample_data

    def forward_propagate(self, X, W, b):

        z = np.dot(X, W) + b

        return z

    def sigmoid(self, z):

        s = 1 / ( 1 + np.exp(-z))

        return s

    def compute_cost(self, Y, Yp):

        cost = - (Y * np.log(Yp) + (1 - Y) * np.log(1 - Yp))

        return cost

    def output_layer_error(self, Y, Yp):

        error = (Y - Yp) * self.sigmoid_derivative(Yp)

        return error

    def hidden_layer_error(self, W, error, node_output):

        if error.shape != ():
            error = error.reshape(len(error), 1)

        layer_error = np.multiply(W, error) * self.sigmoid_derivative(node_output)

        return layer_error

    def sigmoid_derivative(self, s):

        ds = s * (1 - s)

        return ds

    def runner_nn_no_layer(self, predictor, classes, folds_dict):

        """
        Run the procedures for the nn algorithm.
        """

        print('[ INFO ]: Initiate NN runner...')

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
            class_weights = self.train_nn_no_layer(train_set, new_class_columns, 0.075)
            class_results, scores, classification_accuracy = self.test_nn_no_layer(test_set, new_class_columns, class_weights)

            # Determine which tree is the best
            if classification_accuracy > best_classification_accuracy:
                best_fold_class_results = class_results
                best_fold_scores = scores
                best_classification_accuracy = classification_accuracy

        return best_fold_class_results, best_fold_scores, best_classification_accuracy, class_weights

    def runner_nn_one_layer(self, predictor, classes, folds_dict):

        """
        Run the procedures for the nn algorithm.
        """

        print('[ INFO ]: Initiate NN runner...')

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
            class_weights = self.train_nn_one_layer(train_set, new_class_columns, 0.01, 5)
            class_results, scores, classification_accuracy = self.test_nn_one_layer(test_set, new_class_columns, class_weights, 5)

            # Determine which tree is the best
            if classification_accuracy > best_classification_accuracy:
                best_fold_class_results = class_results
                best_fold_scores = scores
                best_classification_accuracy = classification_accuracy

        return best_fold_class_results, best_fold_scores, best_classification_accuracy, class_weights

    def runner_nn_two_layer(self, predictor, classes, folds_dict):

        """
        Run the procedures for the nn algorithm.
        """

        print('[ INFO ]: Initiate NN runner...')

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
            class_weights = self.train_nn_two_layer(train_set, new_class_columns, 0.01, [5,4])
            class_results, scores, classification_accuracy = self.test_nn_two_layer(test_set, new_class_columns, class_weights, [5,4])

            # Determine which tree is the best
            if classification_accuracy > best_classification_accuracy:
                best_fold_class_results = class_results
                best_fold_scores = scores
                best_classification_accuracy = classification_accuracy

        return best_fold_class_results, best_fold_scores, best_classification_accuracy, class_weights
