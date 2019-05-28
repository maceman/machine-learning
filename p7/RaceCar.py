__author__ = 'Mike Macey'

import numpy as np
import pandas as pd


class RaceCar:

    """
    This class holds all the properties that are associated with a car. The car
    can leverage all functions within the class to make it to the finish line
    one the provided race track.
    """

    def __init__(self,
                 state_x,
                 state_y,
                 velocity_x,
                 velocity_y,
                 data,
                 r,
                 c,
                 learning_rate,
                 discount_rate
                 ):
        self.state_x = state_x
        self.state_y = state_y
        self.last_state_x = None
        self.last_state_y = None
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.data = data
        self.r = r
        self.c = c
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.track = None
        self.value = None
        self.iterations = 0
        self.q_matrix = self.init_q_matrix()
        self.value_track = np.random.rand(r, c)
        self.orig_state_x = state_x
        self.orig_state_y = state_y

    # Initiate the state of the car
    def init_state(self):
        self.set_state(self.state_x, self.state_y)

    # Return the current state of the car
    def get_state(self):
        return self.state_x, self.state_y

    # Set the state of the car
    def set_state(self, x, y):
        self.state_x = x
        self.state_y = y
        try:
            self.value = self.track[y][x]
        except:
            self.value = '#'
        try:
            self.track[self.state_y][self.state_x] = 'C'
        except:
            self.restart_new()
            self.track[self.state_y][self.state_x] = 'C'

    # Set the previous state of the car
    def set_last_state(self, x, y):
        self.last_state_x = x
        self.last_state_y = y

    # Move the car in the upward direction
    def drive_up(self):
        self.track[self.state_y][self.state_x] = self.value
        self.set_last_state(self.state_x, self.state_y)
        self.set_state(self.state_x, self.state_y - self.velocity_y)
        self.iterations = self.iterations + 1

    # Move the car in the left direction
    def drive_left(self):
        self.track[self.state_y][self.state_x] = self.value
        self.set_last_state(self.state_x, self.state_y)
        self.set_state(self.state_x - self.velocity_x, self.state_y)
        self.iterations = self.iterations + 1

    # Move the car in the downward direction
    def drive_down(self):
        self.track[self.state_y][self.state_x] = self.value
        self.set_last_state(self.state_x, self.state_y)
        self.set_state(self.state_x, self.state_y + self.velocity_y)
        self.iterations = self.iterations + 1

    # Move the car in the right direction
    def drive_right(self):
        self.track[self.state_y][self.state_x] = self.value
        self.set_last_state(self.state_x, self.state_y)
        self.set_state(self.state_x + self.velocity_x, self.state_y)
        self.iterations = self.iterations + 1

    # Increase the car's acceleration in the Y-direction
    def accelerate_y(self):

        if self.velocity_y < 5:
            if np.random.rand() > 0.2:
                self.velocity_y = self.velocity_y + 1

    # Increase the car's acceleration in the X-direction
    def accelerate_x(self):

        if self.velocity_x < 5:
            if np.random.rand() > 0.2:
                self.velocity_x = self.velocity_x + 1

    # Decelerate the car in the Y-direction
    def decelerate_y(self):

        if self.velocity_y > 0:
            self.velocity_y = self.velocity_y - 1

    # Decelerate the car in the X-direction
    def decelerate_x(self):

        if self.velocity_x > 0:
            self.velocity_x = self.velocity_x - 1

    # Set the car's new position back at the starting line
    def restart_new(self):
        self.set_state(self.orig_state_x, self.orig_state_y)

    # Set the car's new position at its last state before crashing
    def restart_last_position(self):
        self.set_last_state()

    # Make the car's track based on the provided dimensions
    def make_track(self):
        self.track = [None] * self.r
        for i in range(1, len(self.data)):
            self.track[i-1] = list(self.data[i])

    # Print the track to view the car's progress through the environment
    def print_track(self):

        for r in range(len(self.track)):
            print(self.track[r])

    # Print the value track for the value iteration algorithm
    def print_value_track(self):

        for r in range(len(self.value_track)):
            print(self.value_track[r])

    # Print the current X, Y state of the car
    def print_state(self):
        x, y = self.get_state()
        print('X:', x)
        print('Y:', y)

    # Check whether the car can drive up or not
    def can_drive_up(self):

        try:
            x, y = self.get_state()
            y = y - self.velocity_y
            return True
        except:
            return False

    # Check whether the car can drive left or not
    def can_drive_left(self):

        try:
            x, y = self.get_state()
            x = x - self.velocity_x
            return True
        except:
            return False

    # Check whether the car can drive down or not
    def can_drive_down(self):

        try:
            x, y = self.get_state()
            y = y + self.velocity_y
            return True
        except:
            return False

    # Check whether the car can drive right or not
    def can_drive_right(self):

        try:
            x, y = self.get_state()
            x = x + self.velocity_x
            return True
        except:
            return False

    # Check the block up from the car
    def check_up(self):

        block = self.track[self.state_y - 1][self.state_x]
        reward = self.check_reward(block)
        return reward

    # Check the block left from the car
    def check_left(self):

        block = self.track[self.state_y][self.state_x - 1]
        reward = self.check_reward(block)
        return reward

    # Check the block down from the car
    def check_down(self):

        block = self.track[self.state_y + 1][self.state_x]
        reward = self.check_reward(block)
        return reward

    # Check the block right from the car
    def check_right(self):

        block = self.track[self.state_y][self.state_x + 1]
        reward = self.check_reward(block)
        return reward

    # Check the reward for a particular action
    def check_reward(self, block):

        if block == '.':
            return 10
        elif block == '#':
            return -100
        elif block == 'S':
            return 0
        else:
            return 100

    # Check the total reward of a particular state
    def check_total_reward(self):

        up_reward = 0
        left_reward = 0
        down_reward = 0
        right_reward = 0

        up = self.can_drive_up()
        left = self.can_drive_left()
        down = self.can_drive_down()
        right = self.can_drive_right()

        # Check the reward in every direction form the current state
        if up:
            up_reward = self.check_up() + (self.discount_rate * self.value_track[self.state_y - 1][self.state_x])
        if left:
            left_reward = self.check_left() + (self.discount_rate * self.value_track[self.state_y][self.state_x - 1])
        if down:
            down_reward = self.check_down() + (self.discount_rate * self.value_track[self.state_y + 1][self.state_x])
        if right:
            right_reward = self.check_right() + (self.discount_rate * self.value_track[self.state_y][self.state_x + 1])

        return up_reward + left_reward + down_reward + right_reward

    # Carry out the value iteration algorithm for the car object
    def value_iteration(self):

        if self.track[self.state_y][self.state_x] != 'F':

            if self.track[self.state_y][self.state_x] == '#':
                self.restart_last_position()

            up_reward = 0
            left_reward = 0
            down_reward = 0
            right_reward = 0

            up = self.can_drive_up()
            left = self.can_drive_left()
            down = self.can_drive_down()
            right = self.can_drive_right()

            if up:
                up_reward = self.check_up() + (self.discount_rate * self.value_track[self.state_y - 1][self.state_x]) \
                + (self.discount_rate * self.check_up())
            if left:
                left_reward = self.check_left() + (self.discount_rate * self.value_track[self.state_y][self.state_x - 1]) \
                + (self.discount_rate * self.check_left())
            if down:
                down_reward = self.check_down() + (self.discount_rate * self.value_track[self.state_y + 1][self.state_x]) \
                + (self.discount_rate * self.check_down())
            if right:
                right_reward = self.check_right() + (self.discount_rate * self.value_track[self.state_y][self.state_x + 1]) \
                + (self.discount_rate * self.check_right())

            state_reward = 0
            action = None
            rewards = {
                'up': up_reward,
                'left': left_reward,
                'down': down_reward,
                'right': right_reward
            }
            for a, r in rewards.items():
                if state_reward < r:
                    state_reward = r
                    action = a

            if action == 'up':
                self.drive_up()
            elif action == 'left':
                self.drive_left()
            elif action == 'down':
                self.drive_down()
            else:
                self.drive_right()

    # Initialize the Q-learning Q-matrix
    def init_q_matrix(self):

        q_matrix = {
            'S': [0.0,0.0,0.0,0.0],
            '.': [0.0,0.0,0.0,0.0],
            '#': [0.0,0.0,0.0,0.0],
            'F': [0.0,0.0,0.0,0.0],
        }

        return q_matrix

    # Carry out the Q-learning algorithm for the car object
    def q_learning(self):

        if self.track[self.state_y][self.state_x] != 'F':

            if self.track[self.state_y][self.state_x] == '#':
                self.restart_last_position()

            q_matrix = self.init_q_matrix()

            up_reward = 0
            left_reward = 0
            down_reward = 0
            right_reward = 0

            up = self.can_drive_up()
            left = self.can_drive_left()
            down = self.can_drive_down()
            right = self.can_drive_right()

            for k, v in q_matrix.items():

                if up:
                    try:
                        if self.track[self.state_y - 1][self.state_x] == 'k':
                            v[0] = self.check_up()
                    except:
                        continue
                if left:
                    try:
                        if self.track[self.state_y][self.state_x - 1] == 'k':
                            v[1] = self.check_left()
                    except:
                        continue
                if down:
                    try:
                        if self.track[self.state_y + 1][self.state_x] == 'k':
                            v[2] = self.check_down()
                    except:
                        continue
                if right:
                    try:
                        if self.track[self.state_y][self.state_x + 1] == 'k':
                            v[3] = self.check_right()
                    except:
                        continue

            max_action = None
            max_array = []
            for v in q_matrix.values():
                max_array.append(np.argmax(v))

            max_action = np.argmax(max_array)

            if max_action == 0:
                action = 'up'
            elif max_action == 1:
                action = 'left'
            elif max_action == 2:
                action = 'down'
            else:
                action = 'right'

            if action == 'up' and self.can_drive_up():
                self.drive_up()
            elif action == 'left' and self.can_drive_left():
                self.drive_left()
            elif action == 'down' and self.can_drive_down():
                self.drive_down()
            elif action == 'right' and self.can_drive_right():
                self.drive_right()
