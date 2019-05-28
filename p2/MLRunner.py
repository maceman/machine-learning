__author__ =  'Mike Macey'

"""
This class serves as the runner file for the pa2 machine learning clustering program.
"""

# Import necessary packages
import numpy as np
import pandas as pd
from iris import IrisProcessing as ip
from glass import GlassProcessing as gp
from spam import SpamProcessing as sp
from algorithms import Algorithms as alg

# Set display options
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

# Define local paths
glass_path = "/Users/maceyma/Desktop/605.649/pa2/data/glass.data"
iris_path = "/Users/maceyma/Desktop/605.649/pa2/data/iris.data"
spam_path = "/Users/maceyma/Desktop/605.649/pa2/data/spambase.data"
log_path = "/Users/maceyma/Desktop/605.649/pa2/Macey_pa2_Log.txt"

def main():

    # Create logging file
    log = open(log_path,'w')

    print('Program Start \n')
    log.write('Program Start \n')

    # Execute program over the iris dataset
    iris = ip(iris_path)
    iris_selected_features, iris_selected_clusters, iris_basePerformance = iris.iris_runner()
    print(iris_selected_features)
    print(iris_selected_clusters)
    print(iris_basePerformance)
    log.write(str(iris_selected_features) + '\n')
    log.write(str(iris_selected_clusters) + '\n\n')
    log.write('Iris base performance: ' + str(iris_basePerformance) + '\n\n')

    # Execute the program over the glass dataset
    glass = gp(glass_path)
    glass_selected_features, glass_selected_clusters, glass_basePerformance, glass_performance = glass.glass_runner()
    print(glass_selected_features)
    print(glass_selected_clusters)
    print(glass_basePerformance)
    print(glass_performance)
    log.write(str(glass_selected_features) + '\n')
    log.write(str(glass_selected_clusters) + '\n\n')
    log.write('Glass base performance: ' + str(glass_basePerformance) + '\n\n')
    log.write('Glass overall performance: ' + str(glass_performance) + '\n\n')

    # Execute the program over the spam dataset
    spam = sp(spam_path)
    spam_selected_features, spam_selected_clusters, spam_basePerformance, spam_performance = spam.spam_runner()
    print(spam_selected_features)
    print(spam_selected_clusters)
    print(spam_basePerformance)
    print(spam_performance)
    log.write(str(spam_selected_features) + '\n')
    log.write(str(spam_selected_clusters) + '\n\n')
    log.write('Spam base performance: ' + str(spam_basePerformance) + '\n\n')
    log.write('Spam overall performance: ' + str(spam_performance) + '\n\n')

    print('Program End')
    log.write('Program End')

    # Close log
    log.close()


if __name__ == '__main__':
    main()
