__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
import copy
import itertools
import warnings

# Suppres warnings that realte to K-means (Divide by zero, nan values)
np.seterr(divide='ignore', invalid='ignore')

class Algorithms:

    """
    This class serves as a holding ground for the algorithms needed to carry out
    the computations needed to classify the iris, glass and spam datasets into
    clusters.

    Algorithms:

        -K-Means Clustering
        -Silhouette Coefficient Calculation
        -Stepwise-Forward-Selection (SFS) for feature selection
        -Normalization
        -Random Sample

    """


    def __init__(self):
        print("Algorithm object created!")


    def k_means(self, data, feature_set, k):

        """
        Compute the K-means clustering algorithm over the provided dataset, given
        the feature set and value for K clusters.
        """

        print('[ INFO ]: Running K-means over feature set: {}...'.format(feature_set))

        # Convert dataset to array format for easier processing
        feature_df = data[feature_set]
        array_data = np.array(feature_df)

        # Intialize K random centroids
        random_centroids = []

        # Compute number of features
        n_features = len(feature_set)

        # Initialize centroid placeholders
        centroids_updated = np.array(np.random.rand(k, n_features))
        centroids_initial = np.zeros(centroids_updated.shape)

        # Initialize clusters placeholder
        clusters = np.zeros(len(array_data))

        # Initialize centroid error placeholder
        centroid_error = np.linalg.norm(centroids_updated - centroids_initial, axis=None)

        # Loop until centroid error is zero or nan
        while centroid_error != 0.0 and np.isnan(centroid_error) == False:

            for i in range(len(array_data)):

                # Compute the distance between data points and centroids
                cluster_distance = np.linalg.norm(array_data[i] - centroids_updated, axis=1)

                # Find the smallest distance
                cluster = np.argmin(cluster_distance)

                # Assign cluster to data point
                clusters[i] = cluster

            # Reset centroid values
            centroids_initial = copy.deepcopy(centroids_updated)

            for i in range(k):

                # Extract data points related to each cluster
                cluster_members = [array_data[j] for j in range(len(array_data)) if clusters[j] == i]
                if len(cluster_members) != 0:
                    centroids_updated[i] = np.mean(cluster_members, axis=0)

            # Update the error for the new centroids
            centroid_error = np.linalg.norm(centroids_updated - centroids_initial, axis=None)

        # Add computed clusters to data set
        cluster_df = pd.DataFrame(array_data)
        cluster_df['cluster'] = clusters

        return cluster_df


    def silhouette_coefficient(self, clustered_df):

        """
        Compute the silhouette coefficient for the clustered dataset.
        """

        print('[ INFO ]: Calculating the silhouette soefficient...')

        # Initialize silhouette coefficient values
        sil_coefs = np.zeros(len(clustered_df))

        # Loop through each data point in the clustered dataset
        for row in range(len(clustered_df)):

            # Initialize row silhouette coefficients
            row_cluster_sil_coefs = np.zeros((1, 1))
            anti_row_cluster_sil_coefs = np.zeros((1, clustered_df[clustered_df.columns[-1]].nunique() - 1))

            # Initialize cluster counter
            i = 0

            # Loop through each cluster
            for c in clustered_df[clustered_df.columns[-1]].unique():

                # Determine if row belongs to cluster
                if clustered_df[clustered_df.columns[-1]].iloc[row] == c:

                    cluster_points = clustered_df.loc[clustered_df[clustered_df.columns[-1]] == c]

                    # Initialize silhouette coefficient for data point
                    s = 0

                    # Loop through all rows not equal to the row in question
                    for sub_row in range(len(cluster_points)):

                        # Calculate the average distance between the point in question
                        # and all other points in the same cluster
                        if clustered_df.index[row] != cluster_points.index[sub_row]:

                            a = np.array(clustered_df.iloc[row])
                            b = np.array(cluster_points.iloc[sub_row])
                            s = s + np.linalg.norm(a - b)

                    # Compute average silhouette coefficient from cluster points
                    row_cluster_sil_coefs[0][0] = s / len(cluster_points - 1)

                else:

                    cluster_points = clustered_df.loc[clustered_df[clustered_df.columns[-1]] == c]

                    # Initialize silhouette coefficient for data point
                    s = 0

                    # Loop through all rows not equal to the row in question
                    for sub_row in range(len(cluster_points)):

                        # Calculate the average distance between the point in question
                        # and all other points not in the same cluster
                        a = np.array(clustered_df.iloc[row])
                        b = np.array(cluster_points.iloc[sub_row])
                        s = s + np.linalg.norm(a - b)

                    # Compute average silhouette coefficient from non-cluster points
                    anti_row_cluster_sil_coefs[0][i] = s / len(cluster_points)

                    # Increment cluster count
                    i = i + 1

            # Handle division by zero/nan values
            try:
                b = np.min(anti_row_cluster_sil_coefs)
            except:
                b = 0.5

            # Compute silhouette coefficient for data point
            a = row_cluster_sil_coefs[0][0]
            sil_coefs[row] = np.divide((b - a), max(a, b))

        # Compute average silhouette coefficient for dataset
        score = np.sum(sil_coefs) / len(sil_coefs)

        return score


    def stepwise_forward_selection(self, df, features, k):

        """
        Conduct stepwise forward feature selection ovver the provided dataset.
        """

        print('[ INFO ]: Running Stepwise Forward Feature Selction...')

        # Remove classes column
        x = df.drop(df.columns[-1], axis = 1)

        # Normalize the data
        x = self.normalize_data(x)

        # Intitialize performance benchmarks
        basePerformance = -1000000
        bestPerformance = -1000000

        # Initialize best_features variable with feature set
        best_features = features
        selected_clusters = None
        selected_features = []

        # Loop through each feature set size
        for i in range(1,len(features)+1):

            # Loop through each feature set combination
            for f in itertools.combinations(features, i):

                f = np.array(f)

                # See if feature set length is 1 to test features by themself first
                if i == 1:

                    h = self.k_means(x, f, k)
                    currentPerformance = self.silhouette_coefficient(h)
                    print('Performance: ' + str(currentPerformance))

                    # Determine if benchmark measures need to be updated based on performance
                    if currentPerformance > bestPerformance:
                        bestPerformance = currentPerformance
                        best_features = f
                        best_clusters = h

                # Check if the best performing features are in the next set of feature testing
                elif i != 1 and set(best_features).issubset(f):

                    h = self.k_means(x, f, k)
                    currentPerformance = self.silhouette_coefficient(h)
                    print('Performance: ' + str(currentPerformance))

                    # Determine if benchmark measures need to be updated based on performance
                    if currentPerformance > bestPerformance:
                        bestPerformance = currentPerformance
                        best_features = f
                        best_clusters = h

            # Determine if benchmark measures need to be updated based on performance
            if bestPerformance > basePerformance:
                basePerformance = bestPerformance
                selected_features = best_features
                selected_clusters = best_clusters

            # This determines if any new features have been added to the feature set.
            # If not, the loop is broken
            if len(selected_features) == i - 1:
                break

        return selected_features, selected_clusters, basePerformance


    def normalize_data(self, df):

        """
        Normalize the provided dataset for enhanced clustering computations
        """

        print('[ INFO ]: Normalizing data...')

        for col in df.columns:

            col_min = np.min(df[col])
            col_max = np.max(df[col])

            # Normalize data
            df[col] = np.divide(df[col] - col_min, col_max - col_min)

        return df

    def random_feature_sample(self, df, frac):

        """
        Produce a random sample of the provided dataset for enhanced clustering computations
        """

        print('[ INFO ]: Normalizing data...')

        random_sample_data = df.sample(frac=frac, random_state=1)

        return random_sample_data

    def recompute_stepwise_with_population(self, df, features, k):

        clusters = self.k_means(df, features, k)
        performance = self.silhouette_coefficient(clusters)

        return clusters, performance
