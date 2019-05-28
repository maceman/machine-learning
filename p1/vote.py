__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd

class VoteProcessing:

    """

    This class implements a procedure for training and testing two learning algorithms
    (Winnow-2 and Naive Bayes) on the 1984 United States Congressional Voting Record
    data set. This procedure consists of the following steps:

        1. Preprocess the data
        2. Impute any missing values
        3. Randomize the order of each observation in the data
        4. One-hot-encode all features and classes
        5. Split the data into training and testing sets
        6. Train the Winnow-2 algorithm
        7. Test the Winnow-2 classifier
        8. Train the Naive Bayes algorithm
        9. Test the Naive Bayes classifier

    """

    def __init__(self):
        print("VoteProcessing object created!")

    def preprocess_vote(vote_path):

        """
        Preprocess the raw vote data.
        """

        print('[ INFO ]: Preprocessing vote data...')

        # Rename headers of data frame
        vote_data = pd.read_csv(vote_path, header=None)
        vote_data.columns = ['class','handicapped_infant','water_project_cost_sharin','adoption_of_the_budget_resolution',
                    'physician_fee_freeze','el_salvador_aid','religious_groups_in_schools',
                    'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration',
                    'synfuels_corporation_cutback','education_spending','superfund_right_to-sue',
                    'crime','duty_free_exports','export_administration_act_south_africa']

        # Move class column to end of data frame
        vote_data = vote_data[[col for col in vote_data.columns if col != 'class'] + ['class']]

        # Substitute '?' for 'm' which stands for 'maybe'
        vote_data = vote_data.replace('?', 'm')

        # Place classes into list
        classes = vote_data[vote_data.columns[-1]].unique().tolist()

        return vote_data, classes

    def randomize_vote(vote_data):

        """
        Randomize the observations in the raw data
        """

        print('[ INFO ]: Randomizing vote data...')

        np.random.seed(5)
        vote_data['rand'] = np.random.rand(len(vote_data))
        vote_data = vote_data.sort_values(by=['rand']).reset_index()
        vote_data = vote_data.drop(['rand'], axis=1)

        return vote_data

    def one_hot_encode_vote(vote_data, classes):

        """
        One-hot-encode all features and classes in the raw data set
        """

        print('[ INFO ]: One-hot-encoding vote data...')

        for col in vote_data.columns:
            if col not in ['index','class']:
                for u in vote_data[col].unique():

                    # Convert feature values to 1s and 0s
                    vote_data[col + '_' + str(u)] = (vote_data[col] == u).astype(int)

                vote_data = vote_data.drop(col, axis=1)
            if col == 'class':

                # Create new columns for each unique class
                i = 0
                for u in vote_data[col].unique():
                    vote_data['class_{}'.format(u)] = np.where(vote_data[col] == u, 1, 0)
                    classes[i] = 'class_' + str(u)
                    i = i + 1
                vote_data = vote_data.drop(col, axis=1)

        return vote_data, classes

    def train_test_split_vote(vote_data):

        """
        Split the preprocessed raw data into training and testing sets
        """

        print('[ INFO ]: Creating training and testing set for vote data...')

        train_size = round(0.67*len(vote_data))
        vote_data['index'] = vote_data.index.tolist()

        # Set any record with an index less than 2/3 of the number of records
        # in the data frame to the training set
        train_set = vote_data[vote_data['index'] < train_size]
        train_set = train_set.drop('index', axis=1)

        # Assign the next 1/3 to the testing set
        test_set = vote_data[vote_data['index'] >= train_size]
        test_set = test_set.drop('index', axis=1)

        return train_set, test_set

    def train_winnow_2_vote(train_set, classes, alpha):

        """
        Train the Winnow-2 algorithm over the training set
        """

        print('[ INFO ]: Training vote data with Winnow-2 Classifier...')

        # Create weight vector
        train_set_ones = np.ones_like(np.zeros(train_set.drop(classes, axis=1).shape))
        wv = train_set_ones[0]

        weight_vectors = {}

        for vote_class in classes:
            for col in train_set.columns:
                if col not in classes:
                    for i in range(len(train_set)):

                        # Determine if value in row/column location is equivalent to
                        # value in class column
                        if train_set[col].iloc[i] != train_set[vote_class].iloc[i]:

                            # Decrement the associated feature weight by alpha
                            wv[train_set.columns.get_loc(col)] = wv[train_set.columns.get_loc(col)] / 1.01

            weight_vectors[vote_class] = wv

            # Reset weight vector back to 1s for next class
            wv = np.ones_like(wv)

        return weight_vectors

    def test_winnow_2_vote(test_set, classes, weight_vectors, theta):

        """
        Test the Winnow-2 classifier over the testing set
        """

        print('[ INFO ]: Testing vote data with Winnow-2 Classifier...')

        feature_weights = {}
        weighted_sums = {}
        class_results = {}
        scores = {}

        for vote_class in classes:

            # Pull out the true values for each class
            df = test_set.drop(classes, axis=1)
            true_class = test_set[vote_class]

            # Multiply the feature weights by the associated values in the test set
            feature_weights[vote_class] = np.multiply(df, weight_vectors[vote_class])
            weighted_sums[vote_class] = np.sum(feature_weights[vote_class], axis=1)

            # Place the results into a data frame for comparison
            results = pd.concat([weighted_sums[vote_class], true_class], axis=1)
            results.columns = ['weighted_sum', 'true_class']
            results['pred_class'] = np.where(results['weighted_sum'] > theta, 1, 0)
            class_results[vote_class] = results

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            scores[vote_class] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives
            }

        return class_results, scores

    def train_naive_bayes_vote(train_set, classes):

        """
        Train the Naive Bayes algorithm over the training set
        """

        print('[ INFO ]: Training vote data with Naive Bayes Classifier...')

        class_probabilities = {}
        class_feature_probs = {}

        for vote_class in classes:

            feature_true_probs = {}
            feature_false_probs = {}

            # Find the probability that each class is in the training set
            class_probabilities[vote_class] = len(train_set[(train_set[vote_class] == 1)]) / len(train_set)

            # Compute the conditional feature probabilities based on the class probabilities
            # where the class is present
            class_true = train_set[(train_set[vote_class] == 1)]
            for col in class_true.columns:
                if col not in classes:
                    try:
                        true_true = len(class_true[(class_true[col] == 1)]) / len(class_true)
                    except:
                        true_true = 0
                    feature_true_probs[col] = true_true

            # Compute the conditional feature probabilities based on the class probabilities
            # where the class is not present
            class_false = train_set[(train_set[vote_class] == 0)]
            for col in class_false.columns:
                if col not in classes:
                    try:
                        false_false = len(class_false[(class_false[col] == 0)]) / len(class_false)
                    except:
                        false_false = 0
                    feature_false_probs[col] = false_false

            class_feature_probs[vote_class] = [feature_true_probs, feature_false_probs]

        return class_probabilities, class_feature_probs

    def test_naive_bayes_vote(test_set, classes, class_probabilities, class_feature_probs):

        """
        Test the Naive Bayes classifier over the testing set
        """

        print('[ INFO ]: Testing vote data with Naive Bayes Classifier...')

        class_results = {}
        scores = {}

        for vote_class in classes:

            # Create new column for class predictions
            feature_set = test_set.drop(classes, axis=1)
            feature_set['pred_class'] = 0
            true_class = test_set[vote_class]

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
                            true_prob = class_feature_probs[vote_class][0].get(col)
                            false_prob = 1 - class_feature_probs[vote_class][1].get(col)

                        else:

                            # Calculate probabilities assuming the class is not present or 0
                            true_prob = 1 - class_feature_probs[vote_class][0].get(col)
                            false_prob = class_feature_probs[vote_class][1].get(col)

                        # Multiply all feature probabilities together for each record
                        true_probs_sum = true_probs_sum * true_prob
                        false_probs_sum = false_probs_sum * false_prob

                # Multiply class conditional probabilities by conditional feature probabilities
                true_conditional_prob_sum = class_probabilities[vote_class] * true_probs_sum
                false_conditional_prob_sum = (1 - class_probabilities[vote_class]) * false_probs_sum

                # Determine which probability is highest - highest one is selected as the prediction value
                if true_conditional_prob_sum > false_conditional_prob_sum:
                    feature_set['pred_class'].iloc[row] = 1

            # Place the results into a data frame for comparison
            results = pd.concat([feature_set['pred_class'], true_class], axis=1)
            results.columns = ['pred_class', 'true_class']
            class_results[vote_class] = results

            # Calculate the number of TP, TN, FP, FN
            true_positives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 1)])
            true_negatives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 0)])
            false_positives = len(results.loc[(results['true_class'] == 0) & (results['pred_class'] == 1)])
            false_negatives = len(results.loc[(results['true_class'] == 1) & (results['pred_class'] == 0)])

            scores[vote_class] = {
                'TP' : true_positives,
                'TN' : true_negatives,
                'FP' : false_positives,
                'FN' : false_negatives
            }

        return class_results, scores
