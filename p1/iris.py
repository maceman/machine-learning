__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd

class IrisProcessing:

    """

    This class implements a procedure for training and testing two learning algorithms
    (Winnow-2 and Naive Bayes) on the Iris Plants data set. This procedure
    consists of the following steps:

        1. Preprocess the data
        2. Impute any missing values
        3. Randomize the oiris_dataer of each observation in the data
        4. One-hot-encode all features and classes
        5. Split the data into training and testing sets
        6. Train the Winnow-2 algorithm
        7. Test the Winnow-2 classifier
        8. Train the Naive Bayes algorithm
        9. Test the Naive Bayes classifier

    """

    def __init__(self):
        print("IrisProcessing object created!")

    def preprocess_iris(iris_path):

        """
        Preprocess the raw cancer data.
        """

        print('[ INFO ]: Preprocessing iris data...')

        # Rename headers of data frame
        iris_data = pd.read_csv(iris_path, header=None)
        iris_data.columns = ['sepal_length','sepal_width','petal_length','petal_width','iris_class']

        # Place classes into list
        classes = iris_data[iris_data.columns[-1]].unique().tolist()

        return iris_data, classes

    def randomize_iris(iris_data):

        """
        Randomize the observations in the raw data
        """

        print('[ INFO ]: Randomizing iris data...')

        np.random.seed(3)
        iris_data['rand'] = np.random.rand(len(iris_data))
        iris_data = iris_data.sort_values(by=['rand']).reset_index()
        iris_data = iris_data.drop(['rand'], axis=1)

        return iris_data

    def one_hot_encode_iris(iris_data, classes):

        """
        One-hot-encode all features and classes in the raw data set
        """

        print('[ INFO ]: One-hot-encoding iris data...')

        for col in iris_data.columns:
            if col not in ['index','iris_class']:

                # Find mean and std dev of each column
                col_mean = np.mean(iris_data[col])
                col_std  = np.std(iris_data[col])

                # Build out column distribution based on mean and std dev values
                col_dist = [
                    col_mean - (2 * col_std),
                    col_mean-col_std,
                    col_mean,
                    col_mean + col_std,
                    col_mean + (2 * col_std)
                ]

                # Create new columns to support binary features
                iris_data[col + '_less_n2std'] = np.where(iris_data[col] < col_dist[0], 1, 0)
                iris_data[col + '_btw_n2std_n1std'] = np.where(np.logical_and(iris_data[col] < col_dist[1], iris_data[col] >= col_dist[0]), 1, 0)
                iris_data[col + '_btw_n1std_mean'] = np.where(np.logical_and(iris_data[col] < col_dist[2], iris_data[col] >= col_dist[1]), 1, 0)
                iris_data[col + '_btw_mean_p1std'] = np.where(np.logical_and(iris_data[col] < col_dist[3], iris_data[col] >= col_dist[2]), 1, 0)
                iris_data[col + '_btw_p1std_p2std'] = np.where(np.logical_and(iris_data[col] < col_dist[4], iris_data[col] >= col_dist[3]), 1, 0)
                iris_data[col + '_more_p2std'] = np.where(iris_data[col] >= col_dist[4], 1, 0)
                iris_data = iris_data.drop(col, axis=1)
            if col == 'iris_class':

                # Create new columns for each unique class
                for u in iris_data[col].unique():
                    iris_data['{}_class'.format(u)] = np.where(iris_data[col] == u, 1, 0)
                    classes = np.char.replace(classes, u, str(u) + '_class')
                iris_data = iris_data.drop(col, axis=1)

        return iris_data, classes

    def train_test_split_iris(iris_data):

        """
        Split the preprocessed raw data into training and testing sets
        """

        print('[ INFO ]: Creating training and testing set for iris data...')

        train_set_size = round(0.67*len(iris_data))
        iris_data['index'] = iris_data.index.tolist()

        # Set any record with an index less than 2/3 of the number of records
        # in the data frame to the training set
        train_set = iris_data[iris_data['index'] < train_set_size]
        train_set = train_set.drop('index', axis=1)

        # Assign the next 1/3 to the testing set
        test_set = iris_data[iris_data['index'] >= train_set_size]
        test_set = test_set.drop('index', axis=1)

        return train_set, test_set

    def train_winnow_2_iris(train_set, classes, alpha):

        """
        Train the Winnow-2 algorithm over the training set
        """

        print('[ INFO ]: Training iris data with Winnow-2 Classifier...')

        # Create weight vector
        train_set_ones = np.ones_like(np.zeros(train_set.drop(classes, axis=1).shape))
        wv = train_set_ones[0]

        weight_vectors = {}

        for iris_class in classes:
            for col in train_set.columns:
                if col not in classes:
                    for i in range(len(train_set)):

                        # Determine if value in row/column location is equivalent to
                        # value in class column
                        if train_set[col].iloc[i] != train_set[iris_class].iloc[i]:

                            # Decrement the associated feature weight by alpha
                            wv[train_set.columns.get_loc(col)] = wv[train_set.columns.get_loc(col)] / alpha

            weight_vectors[iris_class] = wv

            # Reset weight vector back to 1s for next class
            wv = np.ones_like(wv)

        return weight_vectors

    def test_winnow_2_iris(test_set, classes, weight_vectors, theta):

        """
        Test the Winnow-2 classifier over the testing set
        """

        print('[ INFO ]: Testing iris data with Winnow-2 Classifier...')

        feature_weights = {}
        weighted_sums = {}
        class_results = {}
        scores = {}

        for iris_class in classes:

            # Pull out the true values for each class
            df_dummy = test_set.drop(classes, axis=1)
            true_class = test_set[iris_class]

            # Multiply the feature weights by the associated values in the test set
            feature_weights[iris_class] = np.multiply(df_dummy, weight_vectors[iris_class])
            weighted_sums[iris_class] = np.sum(feature_weights[iris_class], axis=1)

            # Place the results into a data frame for comparison
            results = pd.concat([weighted_sums[iris_class], true_class], axis=1)
            results.columns = ['weighted_sum', 'true_class']
            results['pred_class'] = np.where(results['weighted_sum'] > theta, 1, 0)
            class_results[iris_class] = results

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            scores[iris_class] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives
            }

        return class_results, scores

    def train_naive_bayes_iris(train_set, classes):

        """
        Train the Naive Bayes algorithm over the training set
        """

        print('[ INFO ]: Training iris data with Naive Bayes Classifier...')

        class_probabilities = {}
        class_feature_probs = {}

        for iris_class in classes:

            feature_true_probs = {}
            feature_false_probs = {}

            # Find the probability that each class is in the training set
            class_probabilities[iris_class] = len(train_set[(train_set[iris_class] == 1)]) / len(train_set)

            # Compute the conditional feature probabilities based on the class probabilities
            # where the class is present
            class_true = train_set[(train_set[iris_class] == 1)]
            for col in class_true.columns:
                if col not in classes:
                    try:
                        true_true = len(class_true[(class_true[col] == 1)]) / len(class_true)
                    except:
                        true_true = 0
                    feature_true_probs[col] = true_true

            # Compute the conditional feature probabilities based on the class probabilities
            # where the class is not present
            class_false = train_set[(train_set[iris_class] == 0)]
            for col in class_false.columns:
                if col not in classes:
                    try:
                        false_false = len(class_false[(class_false[col] == 0)]) / len(class_false)
                    except:
                        false_false = 0
                    feature_false_probs[col] = false_false

            class_feature_probs[iris_class] = [feature_true_probs, feature_false_probs]

        return class_probabilities, class_feature_probs

    def test_naive_bayes_iris(test_set, classes, class_probabilities, class_feature_probs):

        """
        Test the Naive Bayes classifier over the testing set
        """

        print('[ INFO ]: Testing iris data with Naive Bayes Classifier...')

        class_results = {}
        scores = {}

        for iris_class in classes:

            # Create new column for class predictions
            feature_set = test_set.drop(classes, axis=1)
            feature_set['pred_class'] = 0
            true_class = test_set[iris_class]

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
                            true_prob = class_feature_probs[iris_class][0].get(col)
                            false_prob = 1 - class_feature_probs[iris_class][1].get(col)

                        else:

                            # Calculate probabilities assuming the class is not present or 0
                            true_prob = 1 - class_feature_probs[iris_class][0].get(col)
                            false_prob = class_feature_probs[iris_class][1].get(col)

                        # Multiply all feature probabilities together for each record
                        true_probs_sum = true_probs_sum * true_prob
                        false_probs_sum = false_probs_sum * false_prob

                # Multiply class conditional probabilities by conditional feature probabilities
                true_conditional_prob_sum = class_probabilities[iris_class] * true_probs_sum
                false_conditional_prob_sum = (1 - class_probabilities[iris_class]) * false_probs_sum

                # Determine which probability is highest - highest one is selected as the prediction value
                if true_conditional_prob_sum > false_conditional_prob_sum:
                    feature_set['pred_class'].iloc[row] = 1

            # Place the results into a data frame for comparison
            results = pd.concat([feature_set['pred_class'], true_class], axis=1)
            results.columns = ['pred_class', 'true_class']
            class_results[iris_class] = results

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            scores[iris_class] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives
            }

        return class_results, scores
