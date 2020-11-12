import os
from maze_env import Maze
import time
import numpy as np
import random

# initiate matrix with dimension from maze
maz = Maze()
dim = maz.MAZE_Limit

np.set_printoptions(suppress=True)
# action options up down left right
action_pool = [(0, 1), (0, -1), (1, 0), (-1, 0)]
action_name = ["right", "left", "down", "up"]

# crate environment
state_value = np.tile(0, dim)
counter = np.tile(0, dim)

# Discounted factor
gamma = 0.9
# Incremental mean
alpha = 0.5
# only TD(lambda) backward view, no TD_N, TD(lambda) forward
lambda_value = 0.5
# [TO-DO] online / exact online / offline (?)
update_type = "online"

n_episode = 2000
nth_reward = -1
for n in range(0, n_episode):
    # random.seed(3)
    episode = []
    action = []
    # Eligibility trace
    et = np.tile(0, dim)
    # starting position
    x = 1
    y = 1
    while not ((x==0 and y==0) or (x==3 and y==3)):
        #### episode starts
        episode.append((x, y))
        #### choose one random action
        i = random.randint(0, 3)
        nth_action = action_pool[i]
        action.append(action_name[i])
        x_next = x + nth_action[0]
        y_next = y + nth_action[1]
        # deal with boundaries
        if x_next > 3:
            x_next = 3
        if y_next > 3:
            y_next = 3
        if x_next < 0:
            x_next = 0
        if y_next < 0:
            y_next = 0
        #### episode ends
        
        # update the eligibility trace
        et = gamma * lambda_value * et
        et[(x, y)] = et[(x, y)] + 1
        # TD Target R_t+1 + gamma * V(S_t+1)
        # TD Error delta_t = td_target - V(S_t)
        td_error = nth_reward + gamma * state_value[(x_next, y_next)] - state_value[(x, y)]
        state_value = state_value + alpha * td_error * et
        # print(np.column_stack((et.round(2), state_value.round(2))), action_name[i])
        
        # update the agent position
        x = x_next
        y = y_next
    episode.append((x, y))
    if (n%20==19): 
        print(n+1)
        print(np.column_stack((et.round(2), state_value.round(2))), len(episode))
    
episode
action
