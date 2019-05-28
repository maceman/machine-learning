__author__ =  'Mike Macey'

"""
This class serves as the runner file for the pa5 machine learning program.
"""

# Import necessary packages
import numpy as np
np.random.seed(1)
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
cancer_path = "/Users/maceyma/Desktop/605.649/p6/Macey_p6/Code/data/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
glass_path = "/Users/maceyma/Desktop/605.649/p6/Macey_p6/Code/data/glass/glass.data"
iris_path = "/Users/maceyma/Desktop/605.649/p6/Macey_p6/Code/data/iris/iris.data"
soy_path = "/Users/maceyma/Desktop/605.649/p6/Macey_p6/Code/data/soybean/soybean-small.data"
vote_path = "/Users/maceyma/Desktop/605.649/p6/Macey_p6/Code/data/vote/house-votes-84.data"
log_path = "/Users/maceyma/Desktop/605.649/p6/Macey_p6/Macey_p6_Log.txt"

def main():

    # Create logging file
    log = open(log_path,'w')
    print('Program Start \n')
    log.write('Program Start \n\n')

    ## CANCER PROCESSING
    cancer = cp(cancer_path)
    c_no_results, c_no_scores, c_no_accuracy, c_no_model, \
    c_one_results, c_one_scores, c_one_accuracy, c_one_model, \
    c_two_results, c_two_scores, c_two_accuracy, c_two_model = cancer.runner()

    ## CANCER LOGGING
    log.write('*** CANCER RESULTS ***\n\n')
    log.write('Cancer - No Layer NN - Results\n\n')
    for cl, data in c_no_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in c_no_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in c_no_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - No Layer NN\n' \
        + str(c_no_accuracy) + '\n\n\n')

    log.write('Cancer - One Layer NN - Results\n\n')
    for cl, data in c_one_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in c_one_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in c_one_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - One Layer NN\n' \
        + str(c_one_accuracy) + '\n\n\n')

    log.write('Cancer - Two Layer NN - Results\n\n')
    for cl, data in c_two_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in c_two_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in c_two_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Two Layer NN\n' \
        + str(c_two_accuracy) + '\n\n\n')

    ## IRIS PROCESSING
    iris = ip(iris_path)
    i_no_results, i_no_scores, i_no_accuracy, i_no_model, \
    i_one_results, i_one_scores, i_one_accuracy, i_one_model, \
    i_two_results, i_two_scores, i_two_accuracy, i_two_model = iris.runner()

    ## IRIS LOGGING
    log.write('*** IRIS RESULTS ***\n\n')
    log.write('Iris - No Layer NN - Results\n\n')
    for cl, data in i_no_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in i_no_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in i_no_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - No Layer NN\n' \
        + str(i_no_accuracy) + '\n\n\n')

    log.write('Iris - One Layer NN - Results\n\n')
    for cl, data in i_one_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in i_one_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in i_one_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - One Layer NN\n' \
        + str(i_one_accuracy) + '\n\n\n')

    log.write('Iris - Two Layer NN - Results\n\n')
    for cl, data in i_two_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in i_two_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in i_two_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Two Layer NN\n' \
        + str(i_two_accuracy) + '\n\n\n')

    ## GLASS PROCESSING
    glass = gp(glass_path)
    g_no_results, g_no_scores, g_no_accuracy, g_no_model, \
    g_one_results, g_one_scores, g_one_accuracy, g_one_model, \
    g_two_results, g_two_scores, g_two_accuracy, g_two_model = glass.runner()

    ## GLASS LOGGING
    log.write('*** GLASS RESULTS ***\n\n')
    log.write('Glass - No Layer NN - Results\n\n')
    for cl, data in g_no_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in g_no_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in g_no_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - No Layer NN\n' \
        + str(g_no_accuracy) + '\n\n\n')

    log.write('Glass - One Layer NN - Results\n\n')
    for cl, data in g_one_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in g_one_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in g_one_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - One Layer NN\n' \
        + str(g_one_accuracy) + '\n\n\n')

    log.write('Glass - Two Layer NN - Results\n\n')
    for cl, data in g_two_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in g_two_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in g_two_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Two Layer NN\n' \
        + str(g_two_accuracy) + '\n\n\n')

    ## SOY PROCESSING
    soy = sp(soy_path)
    s_no_results, s_no_scores, s_no_accuracy, s_no_model, \
    s_one_results, s_one_scores, s_one_accuracy, s_one_model, \
    s_two_results, s_two_scores, s_two_accuracy, s_two_model = soy.runner()

    ## SOY LOGGING
    log.write('*** SOY RESULTS ***\n\n')
    log.write('Soy - No Layer NN - Results\n\n')
    for cl, data in s_no_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in s_no_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in s_no_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - No Layer NN\n' \
        + str(s_no_accuracy) + '\n\n\n')

    log.write('Soy - One Layer NN - Results\n\n')
    for cl, data in s_one_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in s_one_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in s_one_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - One Layer NN\n' \
        + str(s_one_accuracy) + '\n\n\n')

    log.write('Soy - Two Layer NN - Results\n\n')
    for cl, data in s_two_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in s_two_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in s_two_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Two Layer NN\n' \
        + str(s_two_accuracy) + '\n\n\n')

    ## VOTE PROCESSING
    vote = vp(vote_path)
    v_no_results, v_no_scores, v_no_accuracy, v_no_model, \
    v_one_results, v_one_scores, v_one_accuracy, v_one_model, \
    v_two_results, v_two_scores, v_two_accuracy, v_two_model = vote.runner()

    ## VOTE LOGGING
    log.write('*** VOTE RESULTS ***\n\n')
    log.write('Vote - No Layer NN - Results\n\n')
    for cl, data in v_no_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in v_no_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in v_no_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - No Layer NN\n' \
        + str(v_no_accuracy) + '\n\n\n')

    log.write('Vote - One Layer NN - Results\n\n')
    for cl, data in v_one_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in v_one_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in v_one_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - One Layer NN\n' \
        + str(v_one_accuracy) + '\n\n\n')

    log.write('Vote - Two Layer NN - Results\n\n')
    for cl, data in v_two_results.items():
        log.write(cl + '\n')
        log.write(str(data) + '\n\n')
    for cl, score in v_two_scores.items():
        log.write(cl + '\n')
        log.write(str(score) + '\n\n')
    for k, v in v_two_model.items():
        log.write(k + '\n')
        log.write(str(v) + '\n\n')
    log.write('Best Cross-Validated Classification Accuracy - Two Layer NN\n' \
        + str(v_two_accuracy) + '\n\n\n')

    print('Program End')
    log.write('Program End')

    # Close log
    log.close()


if __name__ == '__main__':
    main()
