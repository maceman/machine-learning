__author__ =  'Mike Macey'

"""
This class serves as the runner file for the pa5 machine learning program.
"""

# Import necessary packages
import numpy as np
import pandas as pd
from cancer import CancerProcessing as cp
from glass import GlassProcessing as gp
from iris import IrisProcessing as ip
from soybean import SoyProcessing as sp
from vote import VoteProcessing as vp
from algorithms import Algorithms as alg

# Set display options
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

# Define local paths
cancer_path = "/Users/maceyma/Desktop/605.649/p5/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
glass_path = "/Users/maceyma/Desktop/605.649/p5/data/glass/glass.data"
iris_path = "/Users/maceyma/Desktop/605.649/p5/data/iris/iris.data"
soy_path = "/Users/maceyma/Desktop/605.649/p5/data/soybean/soybean-small.data"
vote_path = "/Users/maceyma/Desktop/605.649/p5/data/vote/house-votes-84.data"
log_path = "/Users/maceyma/Desktop/605.649/p5/Macey_p5_Log.txt"

def main():

    # Create logging file
    log = open(log_path,'w')
    print('Program Start \n')
    log.write('Program Start \n\n')

    ## CANCER PROCESSING
    cancer = cp(cancer_path)
    c_nb_results, c_nb_scores, c_nb_accuracy, \
    c_lr_results, c_lr_scores, c_lr_accuracy, \
    c_ada_results, c_ada_scores, c_ada_accuracy = cancer.runner()

    ## CANCER LOGGING
    log.write('*** CANCER RESULTS ***\n\n')
    log.write('Cancer - Naive Bayes - Results\n\n')
    for cl, data in c_nb_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in c_nb_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Naive Bayes\n' \
        + str(c_nb_accuracy) + '\n\n\n')

    log.write('Cancer - Logistic Regression - Results\n\n')
    for cl, data in c_lr_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in c_lr_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Logistic Regression\n' \
        + str(c_lr_accuracy) + '\n\n\n')

    log.write('Cancer - Adaline - Results\n\n')
    for cl, data in c_ada_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in c_ada_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Adaline\n' \
        + str(c_ada_accuracy) + '\n\n\n')

    ## IRIS PROCESSING
    iris = ip(iris_path)
    i_nb_results, i_nb_scores, i_nb_accuracy, \
    i_lr_results, i_lr_scores, i_lr_accuracy, \
    i_ada_results, i_ada_scores, i_ada_accuracy = iris.runner()

    ## IRIS LOGGING
    log.write('*** IRIS RESULTS ***\n\n')
    log.write('Iris - Naive Bayes - Results\n\n')
    for cl, data in i_nb_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in i_nb_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Naive Bayes\n' \
        + str(i_nb_accuracy) + '\n\n\n')

    log.write('Iris - Logistic Regression - Results\n\n')
    for cl, data in i_lr_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in i_lr_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Logistic Regression\n' \
        + str(i_lr_accuracy) + '\n\n\n')

    log.write('Iris - Adaline - Results\n\n')
    for cl, data in i_ada_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in i_ada_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Adaline\n' \
        + str(i_ada_accuracy) + '\n\n\n')

    ## GLASS PROCESSING
    glass = gp(glass_path)
    g_nb_results, g_nb_scores, g_nb_accuracy, \
    g_lr_results, g_lr_scores, g_lr_accuracy, \
    g_ada_results, g_ada_scores, g_ada_accuracy = glass.runner()

    ## GLASS LOGGING
    log.write('*** GLASS RESULTS ***\n\n')
    log.write('Glass - Naive Bayes - Results\n\n')
    for cl, data in g_nb_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in g_nb_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Naive Bayes\n' \
        + str(g_nb_accuracy) + '\n\n\n')

    log.write('Glass - Logistic Regression - Results\n\n')
    for cl, data in g_lr_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in g_lr_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Logistic Regression\n' \
        + str(g_lr_accuracy) + '\n\n\n')

    log.write('Glass - Adaline - Results\n\n')
    for cl, data in g_ada_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in g_ada_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Adaline\n' \
        + str(g_ada_accuracy) + '\n\n\n')

    ## SOY PROCESSING
    soy = sp(soy_path)
    s_nb_results, s_nb_scores, s_nb_accuracy, \
    s_lr_results, s_lr_scores, s_lr_accuracy, \
    s_ada_results, s_ada_scores, s_ada_accuracy = soy.runner()

    ## SOY LOGGING
    log.write('*** SOY RESULTS ***\n\n')
    log.write('Soy - Naive Bayes - Results\n\n')
    for cl, data in s_nb_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in s_nb_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Naive Bayes\n' \
        + str(s_nb_accuracy) + '\n\n\n')

    log.write('Soy - Logistic Regression - Results\n\n')
    for cl, data in s_lr_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in s_lr_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Logistic Regression\n' \
        + str(s_lr_accuracy) + '\n\n\n')

    log.write('Soy - Adaline - Results\n\n')
    for cl, data in s_ada_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in s_ada_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Adaline\n' \
        + str(s_ada_accuracy) + '\n\n\n')

    ## VOTE PROCESSING
    vote = vp(vote_path)
    v_nb_results, v_nb_scores, v_nb_accuracy, \
    v_lr_results, v_lr_scores, v_lr_accuracy, \
    v_ada_results, v_ada_scores, v_ada_accuracy = vote.runner()

    ## VOTE LOGGING
    log.write('*** VOTE RESULTS ***\n\n')
    log.write('Vote - Naive Bayes - Results\n\n')
    for cl, data in v_nb_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in v_nb_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Naive Bayes\n' \
        + str(v_nb_accuracy) + '\n\n\n')

    log.write('Vote - Logistic Regression - Results\n\n')
    for cl, data in v_lr_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in v_lr_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Logistic Regression\n' \
        + str(v_lr_accuracy) + '\n\n\n')

    log.write('Vote - Adaline - Results\n\n')
    for cl, data in v_ada_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in v_ada_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Adaline\n' \
        + str(v_ada_accuracy) + '\n\n\n')

    print('Program End')
    log.write('Program End')

    # Close log
    log.close()


if __name__ == '__main__':
    main()
