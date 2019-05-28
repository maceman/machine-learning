__author__ = 'Mike Macey'

"""
This class serves as the runner file for the p7 machine learning program.
"""

# Import necessary packages
import numpy as np
#np.random.seed(1)
import pandas as pd
from RaceCar import RaceCar

# Set display options
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

# Define local paths
r_path = "/Users/maceyma/Desktop/605.649/p7/Code/Data/r_track.txt"
l_path = "/Users/maceyma/Desktop/605.649/p7/Code/Data/l_track.txt"
log_path = "/Users/maceyma/Desktop/605.649/p7/Macey_p7/Macey_p7_Log.txt"

def main():

    # Create logging file
    log = open(log_path,'w')
    print('Program Start \n')
    log.write('Program Start \n\n')

    ### Pre-process the R-track in order to set up the proper environment for the agent
    with open(r_path) as f:
        r_data = f.readlines()

    r_data = [x.strip() for x in r_data]
    dim = r_data[0].split(',')

    r_r = int(dim[0])
    r_c = int(dim[1])

    value_matrix = np.zeros((r_r, r_c))

    ### Carry out learning procedure for the R-track
    car_1 = RaceCar(1, 26, 1, 1, r_data, r_r, r_c, 0.75, 0.25)
    car_1.make_track()
    car_1.init_state()
    car_1.print_track()
    for i in range(2000):

        # Run value iteration over the R-track
        car_1.value_iteration()

        if i % 100 == 0:

            print('Track\n')
            car_1.print_track()
            print('Car State: \n')
            car_1.print_state()
            print('Number of actions: ' + str(car_1.iterations))
            log.write(str(car_1.print_track()))
            log.write('Car State: ' + str(car_1.print_state()) + '\n\n')
            log.write('Number of actions: ' + str(car_1.iterations) + '\n\n')

    car_2 = RaceCar(1, 26, 1, 1, r_data, r_r, r_c, 0.75, 0.25)
    car_2.make_track()
    car_2.init_state()
    car_2.print_track()
    for i in range(2000):

        # Run q-learning over the R-track
        car_2.q_learning()

        if i % 100 == 0:

            print('Track\n')
            car_2.print_track()
            print('Car State: \n')
            car_2.print_state()
            print('Number of actions: ' + str(car_2.iterations))
            log.write(str(car_1.print_track()))
            log.write('Car State: ' + str(car_2.print_state()) + '\n\n')
            log.write('Number of actions: ' + str(car_2.iterations) + '\n\n')

    ### Pre-process the L-track in order to set up the proper environment for the agent
    with open(l_path) as f:
        l_data = f.readlines()

    l_data = [x.strip() for x in l_data]
    dim = l_data[0].split(',')

    l_r = int(dim[0])
    l_c = int(dim[1])

    value_matrix = np.zeros((l_r, l_c))

    ### Carry out learning procedure for the R-track
    car_3 = RaceCar(1, 9, 1, 1, l_data, l_r, l_c, 0.75, 0.25)
    car_3.make_track()
    car_3.init_state()
    car_3.print_track()
    for i in range(2000):

        # Run value iteration over the L-track
        car_3.value_iteration()

        if i % 100 == 0:

            print('Track\n')
            car_3.print_track()
            print('Car State: \n')
            car_3.print_state()
            print('Number of actions: ' + str(car_3.iterations))
            log.write(str(car_3.print_track()))
            log.write('Car State: ' + str(car_3.print_state()) + '\n\n')
            log.write('Number of actions: ' + str(car_3.iterations) + '\n\n')

    car_4 = RaceCar(1, 26, 1, 1, r_data, r_r, r_c, 0.75, 0.25)
    car_4.make_track()
    car_4.init_state()
    car_4.print_track()
    for i in range(2000):

        # Run q-learning over the R-track
        car_4.q_learning()

        if i % 100 == 0:
            print('Track\n')
            car_4.print_track()
            print('Car State: \n')
            car_4.print_state()
            print('Number of actions: ' + str(car_4.iterations))
            log.write(str(car_4.print_track()))
            log.write('Car State: ' + str(car_4.print_state()) + '\n\n')
            log.write('Number of actions: ' + str(car_4.iterations) + '\n\n')


    print('Program End')
    log.write('Program End')

    # Close log
    log.close()


if __name__ == '__main__':
    main()
