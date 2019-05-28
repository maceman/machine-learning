__author__ =  'Mike Macey'

"""
This class serves as the runner file for the pa3 machine learning clustering program.
"""

# Import necessary packages
import numpy as np
import pandas as pd
from abalone import AbaloneProcessing as ap
from car import CarProcessing as cp
from image import ImageProcessing as ip
from hardware import HardwareProcessing as hp
from forestfires import ForestfiresProcessing as fp
from wine import WineProcessing as wp
from algorithms import Algorithms as alg

# Set display options
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

# Define local paths
abalone_path = "/Users/maceyma/Desktop/605.649/p4/data/abalone.data"
car_path = "/Users/maceyma/Desktop/605.649/p4/data/car.data"
image_path = "/Users/maceyma/Desktop/605.649/p4/data/segmentation.data"
log_path = "/Users/maceyma/Desktop/605.649/p4/Macey_pa4_Log.txt"

def main():

    # Create logging file
    log = open(log_path,'w')
    print('Program Start \n')
    log.write('Program Start \n')

    abalone = ap(abalone_path)
    a_best_fold_tree, a_score, a_labels, a_pred_labels, a_prune_root, a_prune_score, a_prune_labels, a_prune_pred_labels = abalone.runner()
    abalone_test = alg()
    print('Below is the best tree after cross-validation:\n')
    abalone_test.print_tree(a_best_fold_tree)
    print()
    print('Below is the pruned tree after reduced error pruning:\n')
    abalone_test.print_tree(a_prune_root)
    print('Testing Accuracy\n',a_score,'\n')
    print('Actual Labels\n',a_labels,'\n')
    print('Predicted Labels\n',a_pred_labels,'\n')
    print('Validation Accuracy - Pruned\n',a_prune_score,'\n')
    print('Pruned Actual Labels\n',a_prune_labels,'\n')
    print('Pruned Predicted Labels\n',a_prune_pred_labels,'\n')

    log.write('Testing Accuracy\n' + str(a_score) + '\n')
    log.write('Actual Labels\n' + str(a_labels) + '\n')
    log.write('Predicted Labels\n' + str(a_pred_labels) + '\n')
    log.write('Validation Accuracy - Pruned\n' + str(a_prune_score) + '\n')
    log.write('Pruned Actual Labels\n' + str(a_prune_labels) + '\n')
    log.write('Pruned Predicted Labels\n' + str(a_prune_pred_labels) + '\n')

    car = cp(car_path)
    c_best_fold_tree, c_score, c_labels, c_pred_labels, c_prune_root, c_prune_score, c_prune_labels, c_prune_pred_labels = car.runner()
    car_test = alg()
    print('Below is the best tree after cross-validation:\n')
    car_test.print_tree(c_best_fold_tree)
    print()
    print('Below is the pruned tree after reduced error pruning:\n')
    car_test.print_tree(c_prune_root)
    print('Testing Accuracy\n',c_score,'\n')
    print('Actual Labels\n',c_labels,'\n')
    print('Predicted Labels\n',c_pred_labels,'\n')
    print('Validation Accuracy - Pruned\n',c_prune_score,'\n')
    print('Pruned Actual Labels\n',c_prune_labels,'\n')
    print('Pruned Predicted Labels\n',c_prune_pred_labels,'\n')

    log.write('Testing Accuracy\n' + str(c_score) + '\n')
    log.write('Actual Labels\n' + str(c_labels) + '\n')
    log.write('Predicted Labels\n' + str(c_pred_labels) + '\n')
    log.write('Validation Accuracy - Pruned\n' + str(c_prune_score) + '\n')
    log.write('Pruned Actual Labels\n' + str(c_prune_labels) + '\n')
    log.write('Pruned Predicted Labels\n' + str(c_prune_pred_labels) + '\n')

    image = ip(image_path)
    i_best_fold_tree, i_score, i_labels, i_pred_labels, i_prune_root, i_prune_score, i_prune_labels, i_prune_pred_labels = image.runner()
    image_test = alg()
    print('Below is the best tree after cross-validation:\n')
    image_test.print_tree(i_best_fold_tree)
    print()
    print('Below is the pruned tree after reduced error pruning:\n')
    image_test.print_tree(i_prune_root)
    print('Testing Accuracy\n',i_score,'\n')
    print('Actual Labels\n',i_labels,'\n')
    print('Predicted Labels\n',i_pred_labels,'\n')
    print('Validation Accuracy - Pruned\n',i_prune_score,'\n')
    print('Pruned Actual Labels\n',i_prune_labels,'\n')
    print('Pruned Predicted Labels\n',i_prune_pred_labels,'\n')

    log.write('Testing Accuracy\n' + str(i_score) + '\n')
    log.write('Actual Labels\n' + str(i_labels) + '\n')
    log.write('Predicted Labels\n' + str(i_pred_labels) + '\n')
    log.write('Validation Accuracy - Pruned\n' + str(i_prune_score) + '\n')
    log.write('Pruned Actual Labels\n' + str(i_prune_labels) + '\n')
    log.write('Pruned Predicted Labels\n' + str(i_prune_pred_labels) + '\n')

    print('Program End')
    log.write('Program End')

    # Close log
    log.close()


if __name__ == '__main__':
    main()
