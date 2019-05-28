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

class Node(object):

    def __init__(self):
        self.feature = None
        self.label = None
        self.attribute = None
        self.children = []
        self.n_samples = 0
        self.gain_ratio = None
        self.level = 0
        # print('[ INFO ]: Node object created!')

    def __repr__(self):
        print('Feature:', self.feature)
        print('Label:', self.label)
        print('Parent Attribute:', self.attribute)
        print('Samples:', self.n_samples)
        print('Gain Ratio:', self.gain_ratio)
        print('Level:', self.level,'\n')

class Algorithms:

    """
    This class serves as a holding ground for the algorithms needed to carry out
    the computations needed to score the abalone, car and image segmentation
    datasets.

    Algorithms:

        -prune
        -test
        -test_traversal
        -print_tree
        -cycle
        -build_tree
        -ID3
        -gain_ratio
        -one_hot_encode
        -continuous_to_discrete
        -randomize
        -cross_validation
        -entropy
        -random_feature_sample

    """

    def __init__(self):
        self.root = Node()
        print("[ INFO ]: Algorithm object created!")

    def prune(self, df, predictor, root):

        """
        Traverse through a tree and prune nodes that do not change
        classification accuracy.
        """

        if root.feature == 'Leaf':
            return root

        if root.children == []:
            return root

        # Initialize comparison variables
        most_common_label = None
        n_most_common_label = 0
        node_attribute = None
        correct = 0

        # iterate through each node's children
        for child in root.children:
            if child.n_samples > n_most_common_label:

                #
                most_common_label = child.label
                n_most_common_label = child.n_samples
                node_attribute = child.attribute
                x = df[df[root.feature] == child.attribute]
                correct = correct + len(x)

            # Recursively prune tree
            new_child = self.prune(df, predictor, child)

        # Prune the node if the criteria is met
        if node_attribute != None:
            x = df[df[root.feature] == node_attribute]
            try:
                y = len(x[x[predictor]] == most_common_label)
            except:
                y = 0
            if y > correct:

                # prune node
                root.feature = 'Leaf'
                root.label = most_common_label
                root.children = []
                root.attribute = node_attribute

        return root

    def test(self, df, features, predictor, root):

        """
        Test a given ID3 tree based on a test set.
        """

        print('[ INFO ]: Testing ID3 tree...')

        df['predicted_labels'] = None
        child_df = None

        self.test_traversal(df, child_df, root)
        labels = df['predicted_labels']

        # Determine classification accuracy
        if len(df) != 0:
            score = len(np.where(df[predictor] == df['predicted_labels'])) / len(df)
        else:
            score = 0

        return df, df[predictor], labels, score

    def test_traversal(self, df, child_df, root):

        """
        Traverse the test tree.
        """

        if root.feature == 'Leaf':

            # Create child data frame for node object
            df.loc[child_df.index, 'predicted_labels'] = root.label
        else:
            for child in root.children:
                child_df = df[df[root.feature] == child.attribute]
                if len(child_df) > 0:
                    self.test_traversal(df, child_df, child)

    def print_tree(self, root):

        """
        Print tree such that it can be viewed by the user
        """

        print('[ INFO ]: Printing tree...')

        node_levels = []

        node_list = self.cycle(root, node_levels)

        # Sort nodes by level in the tree
        test = sorted(node_list, key=lambda k: k['level'])

        for i in test:
            for level, node in i.items():
                node.__repr__()

    def cycle(self, root, node_list):

        """
        Traverse through a given tree.
        """

        node_list.append({
            'level' : root.level,
            'root' : root
        })

        if root.children != []:
            for child in root.children:
                self.cycle(child, node_list)

        return node_list


    def build_tree(self, df, features, predictor):

        """
        Build a tree by initializing it.
        """

        print('[ INFO ]: Building ID3 tree...')

        node = self.ID3(df, features, predictor, self.root)

        return node

    def ID3(self, df, features, predictor, root):

        """
        Construct an ID3 tree and recursively traverse each node
        """

        if df[predictor].nunique() == 1:
            root.feature = 'Leaf'
            root.label = df[predictor].unique().tolist()[0]
            root.n_samples = len(df)

            return root

        if len(features) == 0:
            root.feature = 'Leaf'
            root.label = df[predictor].value_counts().index[0]
            root.n_samples = len(df)

            return root

        # Determine which feature is the best feature to split on
        best_feature, gain_ratio = Algorithms.gain_ratio(df, features, predictor)
        root.feature = best_feature
        root.gain_ratio = gain_ratio

        if gain_ratio == 0:
            root.feature = 'Leaf'
            root.label = df[predictor].value_counts().index[0]
            root.n_samples = len(df)
            # print('Label:', root.label)
            return root

        # Loop through each feature's attribute
        for att in df[best_feature].unique().tolist():

            child = Node()
            child_df = df[df[best_feature] == att]
            child_df = child_df.drop([best_feature], axis=1)
            remaining_features = [x for x in features if x != best_feature]
            child.level = root.level + 1
            child.attribute = att
            root.children.append(self.ID3(child_df, remaining_features, predictor, child))
            root.n_samples = len(df)

        return root

    def gain_ratio(df, features, predictor):

        """
        Compute the gain ratio for a set of features
        """

        classes = df[predictor].unique().tolist()

        d_entropy = 0.0
        y = 0
        n = len(df)

        for c in classes:
            x = len(df[df[predictor] == c])
            p_x = x / n
            d_entropy += - p_x * np.log2(p_x)

        f_gain_ratios = []
        for f in features:

            f_att_list = df[f].unique().tolist()
            f_att_dict = []

            f_gain = 0.0
            f_split_info = 0.0

            for f_att in f_att_list:

                f_att_n = len(df[df[f] == f_att])
                p_f_att = f_att_n / n
                f_att_entropy = 0.0

                f_split_info += - p_f_att * np.log2(p_f_att)

                for c in classes:

                    f_x = len(df[(df[f] == f_att) & (df[predictor] == c)])
                    p_f_x = f_x / f_att_n

                    # Handle when p_f_x == 0, set a to arbitrary large negative number
                    if p_f_x == 0:
                        a = -1000
                    else:
                        a = np.log2(p_f_x)

                    f_att_entropy += - p_f_x * a

                f_gain += p_f_att * f_att_entropy

            # Calculate feature entropy
            f_entropy = d_entropy - f_gain

            # Calculate gain ratio
            if f_split_info != 0:
                f_gain_ratio = f_entropy / f_split_info
            else:
                f_gain_ratio = 0.0

            f_gain_ratios.append([f_gain_ratio, f])

        f_gain_ratios = sorted(f_gain_ratios, reverse=True)

        return f_gain_ratios[0][1], f_gain_ratios[0][0]

    def one_hot_encode(self, df, categorical_data):

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

        np.random.seed(1)
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

    def entropy(n, labels):

        """
        Compute the entropy for a set of labels
        """

        ent = 0
        for label in labels.keys():
            p_x = labels[label] / n
            ent += - p_x * math.log(p_x, 2)
        return ent

    def random_feature_sample(self, df, frac):

        """
        Produce a random sample of the provided dataset for enhanced clustering computations
        """

        print('[ INFO ]: Normalizing data...')

        random_sample_data = df.sample(frac=frac, random_state=1)

        return random_sample_data
