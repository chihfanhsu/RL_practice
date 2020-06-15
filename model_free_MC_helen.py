import os
from maze_env import Maze
import time
import numpy as np
import random

maz = Maze()
# only for interaction:
# maz.render()

# initiate matrix with dimension from maze
dim = maz.MAZE_Limit

np.set_printoptions(suppress=True)
# action options up down left right
a = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# crate environment
state_value = np.tile(0, dim)
counter = np.tile(0, dim)

# [DONE] First or Every visit state value counted
visit = "first" # "first" or "every"
# [DONE] Discounted factor
gamma = 0.9
# [DONE] MC
# [TO-DO] if there's block
# [TO-DO] if random start

n_episode = 10000
for n in range(0, n_episode):
    # random.seed(3)
    episode = []
    action = []
    reward_map = np.tile(0, dim)
    td_target = np.tile(0, dim)
    nth_counter = np.tile(0, dim)
    # starting position
    x = 1
    y = 1
    # create one episode
    while not ((x==0 and y==0) or (x==3 and y==3)):
        # print(x, y)
        episode.append((x, y))
        reward_map[(x, y)] = 1
        # [TO-DO] if there are different rewards (immediate or final), then the reward chain needs to be retrieved end-to-begin.
        if (visit == "first"):
            # estimated return:
            # td_target =           R_t+1 + gamma * V(S_t+1)
            td_target = (-1) * reward_map + gamma * td_target
            nth_counter = reward_map
        else: #(every)
            nth_counter[(x, y)] = nth_counter[(x, y)] + 1
            reward_map_every = reward_map.copy()
            reward_map_every[(x, y)] = reward_map_every[(x, y)] + 1
            # estimated return:
            # td_target =                 R_t+1 + gamma * V(S_t+1)
            td_target = (-1) * reward_map_every + gamma * td_target
            
        # print(np.column_stack((nth_counter, td_target.round(2))), visit)
        nth_action = a[random.randint(0, 3)]
        action.append(nth_action)
        # update state position
        x = x + nth_action[0]
        y = y + nth_action[1]
        # deal with boundaries
        if x > 3:
            x = 3
        if y > 3:
            y = 3
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        
    # episode-end
    # print(x, y)
    episode.append((x, y))
    # print(n, len(episode))
    # episode
    # action
    counter = counter + nth_counter
    state_value = state_value + td_target
    # print(np.column_stack((counter, state_value.round(2))))
    
# Here alpha is 1/N after iteration ends
alpha = np.nan_to_num(1 / counter, nan = 0)
final_state_value = np.nan_to_num(state_value * alpha, nan = 0)
print(np.column_stack((counter, final_state_value.round(2))), visit)
# n10000 = final_state_value
# c10000 = counter
# ne10000 = final_state_value
# ce10000 = counter
# print(np.column_stack((c10000, n10000.round(2))))
# print(np.column_stack((ce10000, ne10000.round(2))))
