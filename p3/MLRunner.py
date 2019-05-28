__author__ =  'Mike Macey'

"""
This class serves as the runner file for the pa3 machine learning clustering program.
"""

# Import necessary packages
import numpy as np
import pandas as pd
from ecoli import EcoliProcessing as ep
from forestfires import ForestfiresProcessing as fp
from machine import MachineProcessing as mp
from segmentation import SegmentationProcessing as sp
from algorithms import Algorithms as alg

# Set display options
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

# Define local paths
ecoli_path = "/Users/maceyma/Desktop/605.649/pa3/Macey_pa3/Macey_pa3_code/data/ecoli.data"
ff_path = "/Users/maceyma/Desktop/605.649/pa3/Macey_pa3/Macey_pa3_code/data/forestfires.data"
mach_path = "/Users/maceyma/Desktop/605.649/pa3/Macey_pa3/Macey_pa3_code/data/machine.data"
seg_path = "/Users/maceyma/Desktop/605.649/pa3/Macey_pa3/Macey_pa3_code/data/segmentation.data"
log_path = "/Users/maceyma/Desktop/605.649/pa3/Macey_pa3_Log.txt"

def main():

    # # Create logging file
    log = open(log_path,'w')

    print('Program Start \n')
    log.write('Program Start \n')

    # Execute program over the ecoli dataset
    print('[ INFO ]: Start Ecoli Processing\n')
    log.write('[ INFO ]: Start Ecoli Processing\n')

    ecoli = ep(ecoli_path)
    e_labels, e_score, e_k_value, e_cond_labels, e_cond_score, e_cond_k_value = ecoli.ecoli_runner()

    print('[ INFO ]: Optimal KNN Labels\n' + str(e_labels) + '\n')
    print('[ INFO ]: Optimal KNN Classification Accuracy\n' + str(e_score) + '\n')
    print('[ INFO ]: Optimal KNN K Value\n' + str(e_k_value) + '\n')
    print('[ INFO ]: Optimal Condensed KNN Labels\n' + str(e_cond_labels) + '\n')
    print('[ INFO ]: Optimal Condensed KNN Classification Accuracy\n' + str(e_cond_score) + '\n')
    print('[ INFO ]: Optimal Condensed KNN K Value\n' + str(e_cond_k_value) + '\n')

    log.write('[ INFO ]: Optimal KNN Labels\n' + str(e_labels) + '\n')
    log.write('[ INFO ]: Optimal KNN Classification Accuracy\n' + str(e_score) + '\n')
    log.write('[ INFO ]: Optimal KNN K Value\n' + str(e_k_value) + '\n')
    log.write('[ INFO ]: Optimal Condensed KNN Labels\n' + str(e_cond_labels) + '\n')
    log.write('[ INFO ]: Optimal Condensed KNN Classification Accuracy\n' + str(e_cond_score) + '\n')
    log.write('[ INFO ]: Optimal Condensed KNN K Value\n' + str(e_cond_k_value) + '\n')

    # Execute program over the forestfires dataset
    print('\n[ INFO ]: Start Forestfires Processing\n')
    log.write('\n[ INFO ]: Start Forestfires Processing\n')

    ff = fp(ff_path)
    f_labels, f_score, f_k_value = ff.forestfires_runner()

    print('[ INFO ]: Optimal KNN Values\n' + str(f_labels) + '\n')
    print('[ INFO ]: Optimal KNN RMSE\n' + str(f_score) + '\n')
    print('[ INFO ]: Optimal KNN K Value\n' + str(f_k_value) + '\n')

    log.write('[ INFO ]: Optimal KNN Values\n' + str(f_labels) + '\n')
    log.write('[ INFO ]: Optimal KNN RMSE\n' + str(f_score) + '\n')
    log.write('[ INFO ]: Optimal KNN K Value\n' + str(f_k_value) + '\n')

    # Execute program over the machine dataset
    print('\n[ INFO ]: Start Machine Processing\n')
    log.write('\n[ INFO ]: Start Machine Processing\n')

    mach = mp(mach_path)
    m_labels, m_score, m_k_value = mach.machine_runner()

    print('[ INFO ]: Optimal KNN Values\n' + str(m_labels) + '\n')
    print('[ INFO ]: Optimal KNN RMSE\n' + str(m_score) + '\n')
    print('[ INFO ]: Optimal KNN K Value\n' + str(m_k_value) + '\n')

    log.write('[ INFO ]: Optimal KNN Values\n' + str(m_labels) + '\n')
    log.write('[ INFO ]: Optimal KNN RMSE\n' + str(m_score) + '\n')
    log.write('[ INFO ]: Optimal KNN K Value\n' + str(m_k_value) + '\n')

    # Execute program over the segmentation dataset
    print('\n[ INFO ]: Start Segmentation Processing\n')
    log.write('\n[ INFO ]: Start Segmentation Processing\n')

    seg = sp(seg_path)
    s_labels, s_score, s_k_value, s_cond_labels, s_cond_score, s_cond_k_value = seg.segmentation_runner()

    print('[ INFO ]: Optimal KNN Labels\n' + str(s_labels) + '\n')
    print('[ INFO ]: Optimal KNN Classification Accuracy\n' + str(s_score) + '\n')
    print('[ INFO ]: Optimal KNN K Value\n' + str(s_k_value) + '\n')
    print('[ INFO ]: Optimal Condensed KNN Labels\n' + str(s_cond_labels) + '\n')
    print('[ INFO ]: Optimal Condensed KNN Classification Accuracy\n' + str(s_cond_score) + '\n')
    print('[ INFO ]: Optimal Condensed KNN K Value\n' + str(s_cond_k_value) + '\n')

    log.write('[ INFO ]: Optimal KNN Labels\n' + str(s_labels) + '\n')
    log.write('[ INFO ]: Optimal KNN Classification Accuracy\n' + str(s_score) + '\n')
    log.write('[ INFO ]: Optimal KNN K Value\n' + str(s_k_value) + '\n')
    log.write('[ INFO ]: Optimal Condensed KNN Labels\n' + str(s_cond_labels) + '\n')
    log.write('[ INFO ]: Optimal Condensed KNN Classification Accuracy\n' + str(s_cond_score) + '\n')
    log.write('[ INFO ]: Optimal Condensed KNN K Value\n' + str(s_cond_k_value) + '\n')

    print('Program End')
    log.write('Program End')

    # Close log
    log.close()


if __name__ == '__main__':
    main()
