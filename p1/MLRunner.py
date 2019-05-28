__author__ =  'Mike Macey'

"""
Description
"""

# Import necessary packages
import numpy as np
import pandas as pd
from cancer import CancerProcessing as cp
from glass import GlassProcessing as gp
from iris import IrisProcessing as ip
from soy import SoyProcessing as sp
from vote import VoteProcessing as vp

# Set display options
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

# Define local paths
cancer_path = "/Users/maceyma/Desktop/605.649/pa1/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
glass_path = "/Users/maceyma/Desktop/605.649/pa1/data/glass/glass.data"
iris_path = "/Users/maceyma/Desktop/605.649/pa1/data/iris/iris.data"
soy_path = "/Users/maceyma/Desktop/605.649/pa1/data/soybean/soybean-small.data"
vote_path = "/Users/maceyma/Desktop/605.649/pa1/data/vote/house-votes-84.data"
log_path = "/Users/maceyma/Desktop/605.649/pa1/Macey_pa1_Log.txt"

def main():

    # Create logging file
    log = open(log_path,'w')

    print('Program Start \n')
    log.write('Program Start \n')

    """
    Compute Winnow-2 and Naive Bayes on the Cancer data
    """

    cancer_data, cancer_classes = cp.preprocess_cancer(cancer_path)
    cancer_data = cp.impute_cancer(cancer_data)
    cancer_data = cp.randomize_cancer(cancer_data)
    cancer_data, cancer_classes = cp.one_hot_encode_cancer(cancer_data, cancer_classes)
    cancer_train_set, cancer_test_set = cp.train_test_split_cancer(cancer_data)
    cancer_weight_vector = cp.train_winnow_2_cancer(cancer_train_set, cancer_classes, 1.01) # set alpha
    cancer_winnow_class_results, cancer_winnow_scores = cp.test_winnow_2_cancer(cancer_test_set, cancer_classes, cancer_weight_vector, 1.30) # set theta
    cancer_class_probabilities, cancer_class_feature_probs = cp.train_naive_bayes_cancer(cancer_train_set, cancer_classes)
    cancer_naive_bayes_class_results, cancer_naive_bayes_scores = cp.test_naive_bayes_cancer(cancer_test_set, cancer_classes, cancer_class_probabilities, cancer_class_feature_probs)

    # Print cancer results to console and log
    print('[ INFO ]: Winnow-2 Class Predictions: ' + str(cancer_winnow_class_results))
    print('[ INFO ]: Winnow-2 Scores: ' + str(cancer_winnow_scores))
    log.write('[ INFO ]: Winnow-2 Class Predictions: ' + str(cancer_winnow_class_results))
    log.write('[ INFO ]: Winnow-2 Class Scores: ' + str(cancer_winnow_scores))
    print('[ INFO ]: Naive Bayes Class Predictions: ' + str(cancer_naive_bayes_class_results))
    print('[ INFO ]: Naive Bayes Scores: ' + str(cancer_naive_bayes_scores))
    log.write('[ INFO ]: Naive Bayes Class Predictions: ' + str(cancer_naive_bayes_class_results))
    log.write('[ INFO ]: Naive Bayes Scores: ' + str(cancer_naive_bayes_scores))


    """
    Compute Winnow-2 and Naive Bayes on the Glass data
    """

    glass_data, glass_classes = gp.preprocess_glass(glass_path)
    glass_data = gp.randomize_glass(glass_data)
    glass_data, glass_classes = gp.one_hot_encode_glass(glass_data, glass_classes)
    glass_train_set, glass_test_set = gp.train_test_split_glass(glass_data)
    glass_weight_vector = gp.train_winnow_2_glass(glass_train_set, glass_classes, 1.01) # set alpha
    glass_winnow_class_results, glass_winnow_scores = gp.test_winnow_2_glass(glass_test_set, glass_classes, glass_weight_vector, 5.80) # set theta
    glass_class_probabilities, glass_class_feature_probs = gp.train_naive_bayes_glass(glass_train_set, glass_classes)
    glass_naive_bayes_class_results, glass_naive_bayes_scores = gp.test_naive_bayes_glass(glass_test_set, glass_classes, glass_class_probabilities, glass_class_feature_probs)

    # Print glass results to console and log
    print('[ INFO ]: Winnow-2 Class Predictions: ' + str(glass_winnow_class_results))
    print('[ INFO ]: Winnow-2 Scores: ' + str(glass_winnow_scores))
    log.write('[ INFO ]: Winnow-2 Class Predictions: ' + str(glass_winnow_class_results))
    log.write('[ INFO ]: Winnow-2 Class Scores: ' + str(glass_winnow_scores))
    print('[ INFO ]: Naive Bayes Class Predictions: ' + str(glass_naive_bayes_class_results))
    print('[ INFO ]: Naive Bayes Scores: ' + str(glass_naive_bayes_scores))
    log.write('[ INFO ]: Naive Bayes Class Predictions: ' + str(glass_naive_bayes_class_results))
    log.write('[ INFO ]: Naive Bayes Scores: ' + str(glass_naive_bayes_scores))


    """
    Compute Winnow-2 and Naive Bayes on the Iris data
    """

    iris_data, iris_classes = ip.preprocess_iris(iris_path)
    iris_data = ip.randomize_iris(iris_data)
    iris_data, iris_classes = ip.one_hot_encode_iris(iris_data, iris_classes)
    iris_train_set, iris_test_set = ip.train_test_split_iris(iris_data)
    iris_weight_vector = ip.train_winnow_2_iris(iris_train_set, iris_classes, 1.01) # set alpha
    iris_winnow_class_results, iris_winnow_scores = ip.test_winnow_2_iris(iris_test_set, iris_classes, iris_weight_vector, 2.75) # set theta
    iris_class_probabilities, iris_class_feature_probs = ip.train_naive_bayes_iris(iris_train_set, iris_classes)
    iris_naive_bayes_class_results, iris_naive_bayes_scores = ip.test_naive_bayes_iris(iris_test_set, iris_classes, iris_class_probabilities, iris_class_feature_probs)

    # Print iris results to console and log
    print('[ INFO ]: Winnow-2 Class Predictions: ' + str(iris_winnow_class_results))
    print('[ INFO ]: Winnow-2 Scores: ' + str(iris_winnow_scores))
    log.write('[ INFO ]: Winnow-2 Class Predictions: ' + str(iris_winnow_class_results))
    log.write('[ INFO ]: Winnow-2 Class Scores: ' + str(iris_winnow_scores))
    print('[ INFO ]: Naive Bayes Class Predictions: ' + str(iris_naive_bayes_class_results))
    print('[ INFO ]: Naive Bayes Scores: ' + str(iris_naive_bayes_scores))
    log.write('[ INFO ]: Naive Bayes Class Predictions: ' + str(iris_naive_bayes_class_results))
    log.write('[ INFO ]: Naive Bayes Scores: ' + str(iris_naive_bayes_scores))


    """
    Compute Winnow-2 and Naive Bayes on the Soy data
    """

    soy_data, soy_classes = sp.preprocess_soy(soy_path)
    soy_data = sp.randomize_soy(soy_data)
    soy_data, soy_classes = sp.one_hot_encode_soy(soy_data, soy_classes)
    soy_train_set, soy_test_set = sp.train_test_split_soy(soy_data)
    soy_weight_vector = sp.train_winnow_2_soy(soy_train_set, soy_classes, 1.01) # set alpha
    soy_winnow_class_results, soy_winnow_scores = sp.test_winnow_2_soy(soy_test_set, soy_classes, soy_weight_vector, 29.30) # set theta
    soy_class_probabilities, soy_class_feature_probs = sp.train_naive_bayes_soy(soy_train_set, soy_classes)
    soy_naive_bayes_class_results, soy_naive_bayes_scores = sp.test_naive_bayes_soy(soy_test_set, soy_classes, soy_class_probabilities, soy_class_feature_probs)

    # Print soy results to console and log
    print('[ INFO ]: Winnow-2 Class Predictions: ' + str(soy_winnow_class_results))
    print('[ INFO ]: Winnow-2 Scores: ' + str(soy_winnow_scores))
    log.write('[ INFO ]: Winnow-2 Class Predictions: ' + str(soy_winnow_class_results))
    log.write('[ INFO ]: Winnow-2 Class Scores: ' + str(soy_winnow_scores))
    print('[ INFO ]: Naive Bayes Class Predictions: ' + str(soy_naive_bayes_class_results))
    print('[ INFO ]: Naive Bayes Scores: ' + str(soy_naive_bayes_scores))
    log.write('[ INFO ]: Naive Bayes Class Predictions: ' + str(soy_naive_bayes_class_results))
    log.write('[ INFO ]: Naive Bayes Scores: ' + str(soy_naive_bayes_scores))


    """
    Compute Winnow-2 and Naive Bayes on the Vote data
    """

    vote_data, vote_classes = vp.preprocess_vote(vote_path)
    vote_data = vp.randomize_vote(vote_data)
    vote_data, vote_classes = vp.one_hot_encode_vote(vote_data, vote_classes)
    vote_train_set, vote_test_set = vp.train_test_split_vote(vote_data)
    vote_weight_vector = vp.train_winnow_2_vote(vote_train_set, vote_classes, 1.01) # set alpha
    vote_winnow_class_results, vote_winnow_scores = vp.test_winnow_2_vote(vote_test_set, vote_classes, vote_weight_vector, 4.65) # set theta
    vote_class_probabilities, vote_class_feature_probs = vp.train_naive_bayes_vote(vote_train_set, vote_classes)
    vote_naive_bayes_class_results, vote_naive_bayes_scores = vp.test_naive_bayes_vote(vote_test_set, vote_classes, vote_class_probabilities, vote_class_feature_probs)

    # Print vote results to console and log
    print('[ INFO ]: Winnow-2 Class Predictions: ' + str(vote_winnow_class_results))
    print('[ INFO ]: Winnow-2 Scores: ' + str(vote_winnow_scores))
    log.write('[ INFO ]: Winnow-2 Class Predictions: ' + str(vote_winnow_class_results))
    log.write('[ INFO ]: Winnow-2 Class Scores: ' + str(vote_winnow_scores))
    print('[ INFO ]: Naive Bayes Class Predictions: ' + str(vote_naive_bayes_class_results))
    print('[ INFO ]: Naive Bayes Scores: ' + str(vote_naive_bayes_scores))
    log.write('[ INFO ]: Naive Bayes Class Predictions: ' + str(vote_naive_bayes_class_results))
    log.write('[ INFO ]: Naive Bayes Scores: ' + str(vote_naive_bayes_scores))

    print('Program End')
    log.write('Program End')

    # Close log
    log.close()

if __name__ == '__main__':
    main()
