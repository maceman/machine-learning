__author__ =  'Mike Macey'

# Import necessary packages
import numpy as np
import pandas as pd
from algorithms import Algorithms as alg

class IrisProcessing:

    """
    This class carries out the necessary instructions for processing the iris dataset.
    """

    def __init__(self, iris_path):
        self.iris_path = iris_path
        print("IrisProcessing object created!")

    def preprocess_iris(self):

        """
        Preprocess the raw iris data.
        """

        print('[ INFO ]: Preprocessing iris data...')

        # Rename headers of data frame
        iris_data = pd.read_csv(self.iris_path, header=None)
        iris_data.columns = ['sepal_length','sepal_width','petal_length','petal_width', 'iris_class']

        df_columns = [iris_data.columns[j] for j in range(len(iris_data.columns)) if iris_data.columns[j] != 'iris_class']

        # Place classes into list
        classes = iris_data['iris_class'].unique().tolist()

        return iris_data, df_columns, classes

    def iris_runner(self):

        """
        Execute the program runner over the iris dataset.
        """

        print('[ INFO ]: Initializing the iris program runner...')

        data, features, classes = self.preprocess_iris()
        iris = alg()
        selected_features, selected_clusters, basePerformance = iris.stepwise_forward_selection(data, features, len(classes))

        return selected_features, selected_clusters, basePerformance
