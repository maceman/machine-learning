__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

class Algorithms:

    """
    This class serves as a holding ground for the algorithms needed to carry out
    the computations needed to score the ecoli, forestfires, machine and segmentation
    datasets.

    Algorithms:

        -Condensed K-Nearest Neighbor
        -K-Nearest Neighbor
        -One-Hot-Encode
        -randomize
        -Cross-Validation

    """


    def __init__(self):
        print("[ INFO ]: Algorithm object created!")

    def condensed_knn(self, train_set, var):

        print('[ INFO ]: Running Condensed KNN...')

        # Construct the training set
        train_set = self.randomize(train_set)
        x_train = np.array(train_set.drop(columns=[var]))
        y_train = np.array(train_set[var])

        # Construct Z and initialize values
        z_data = np.zeros((1, len(x_train[0])))
        z_label = np.zeros((1, 1))
        z_len_initial = 0
        z_len_updated = 1000000

        indices = []

        # Loop until no more values are added to Z
        while z_len_updated != z_len_initial:

            z_len_initial = len(z_data)

            # Loop through each xi in X
            for i in range(len(x_train)):

                # Handle the first case of the loop by adding the first record to Z
                if i == 0 and len(z_data) == 1:

                    z_data = x_train[0,:]
                    z_label = y_train[0]
                    indices.append(i)

                    # Remove data point from X
                    z_data = np.array(z_data.reshape((1,len(z_data))))

                else:

                    distances = []

                    for j in range(len(z_data)):

                        # Calcualte the difference between each xi and zi
                        distance = np.linalg.norm(x_train[i,:] - z_data[j,:])
                        distances.append([distance, j])

                    # Sort the first elements of each sub-array: distance
                    distances = sorted(distances)

                    # Find the closest neighbor
                    z_index = distances[0][1]
                    z_lab = z_label[z_index]

                    # Check if classes match
                    if z_lab != y_train[i]:

                        # Append data point to Z
                        z_data = np.vstack((z_data, x_train[i,:]))
                        z_label = np.vstack((z_label, y_train[i]))
                        indices.append(i)

            # Remove data points from X
            x_train = np.array([x_train[x] for x in range(len(x_train)) if x not in indices])
            y_train = np.array([y_train[x] for x in range(len(y_train)) if x not in indices])

            z_len_updated = len(z_data)

        # Construct new training set for regular KNN
        new_train_set = pd.DataFrame(np.hstack((z_data, z_label)))
        new_train_set.columns = [[x for x in train_set.columns if x != var] + [var]]
        new_train_set = new_train_set.apply(pd.to_numeric, errors='ignore')

        print('[ INFO ]: Finishing Condensed KNN...')

        return new_train_set

    def knn(self, train_set, test_set, var, type, k):

        print('[ INFO ]: Running KNN...')

        # Construct the training set
        y_train = np.array(train_set[var])
        x_train = np.array(train_set.drop(columns=[var], index=1))
        x_test = np.array(test_set.drop(columns=[var], index=1))

        labels = []

        # Loop through each xi in the test set
        for i in range(len(x_test)):

            distances = []

            for j in range(len(x_train)):

                # Calculate the differences between each data point
                distance = np.linalg.norm(x_test[i,:] - x_train[j,:])
                distances.append([distance, j])

            # Sort the first elements of each sub-array: distance
            distances = sorted(distances)

            # Find the k-nearest neighbors
            k_labels = []
            for l in range(k):
                k_index = distances[l][1]
                k_labels.append(y_train[k_index])

            # Find mean if prediction type is regression
            if type == 'regression':
                labels.append(np.mean(k_labels))

            # Find most present class if classification
            else:
                label, counts = np.unique(k_labels, return_counts=True)
                max_count = np.argmax(counts)
                labels.append(label[max_count])

        x = pd.DataFrame(x_test)
        x['true_labels'] = test_set[var]
        x['pred_labels'] = labels

        labels = x[['true_labels', 'pred_labels']]

        # Calcualte RMSE
        if type == 'regression':
            score = np.sqrt(np.divide(np.sum(np.power(labels['pred_labels'] - labels['true_labels'], 2)), len(labels)))

        # Calculate classification accuracy
        else:
            results = np.where(x['true_labels'] == x['pred_labels'], 1, 0)
            score = np.count_nonzero(results) / len(results)

        print('[ INFO ]: Finishing KNN...')

        return labels, score

    def one_hot_encode(self, df, categorical_data):

        print('[ INFO ]: Conducting One-Hot-Encoding...')

        for col in df.columns:
            if col in categorical_data:
                for value in df[col].unique():
                    df[str(col) + '_' + str(value)] = np.where(df[col] == value, 1, 0)
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

        print('[ INFO ]: Running {}-fold cross-validation...'.format(folds))

        folds_dict = {}
        for fold in range(folds):
            folds_dict['fold_' + str(fold + 1)] = pd.DataFrame()

        # If prediction type is regression, randomly split dataset into n folds
        if type == 'regression':

            df = self.randomize(df)
            fold_len = round(len(df) / folds)
            #print(df)
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
