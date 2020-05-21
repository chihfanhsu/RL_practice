import os
from maze_env import Maze
from utility import Prediction
import time
import numpy as np

s = (4, 4)
# crate environment
env = np.tile(1, s)
# terminate state
env[0, 0] = 0
env[3, 3] = 0
# block state
env[2, 2] = 0
env

# reward
r = -1
p0 = np.tile(r, s) * env

# action
# up
p1 = p0.copy()
# down
p2 = p0.copy()
# right
p3 = p0.copy()
# left
p4 = p0.copy()

# initialize value
value = np.zeros(s)
for i in range(10):
    if i==0:
        print(i)
        print(value)
        # define next state (if outside of box (wall) -> back to previous state)
        b = p1[2+1, 2] # keep the state where next state will be block.
        p1 = p1[(0, 0, 1, 2), :] * env # up
        p1[2+1, 2] = b
        
        b = p2[2-1, 2] # keep the state where next state will be block.
        p2 = p2[(1, 2, 3, 3), :] * env # down
        p2[2-1, 2] = b
        
        b = p3[2, 2+1] # keep the state where next state will be block.
        p3 = p3[:, (0, 0, 1, 2)] * env # left
        p3[2, 2+1] = b
        
        b = p4[2, 2-1] # keep the state where next state will be block.
        p4 = p4[:, (1, 2, 3, 3)] * env # right
        p4[2, 2-1] = b
    else:
        # define next state (if outside of box (wall) -> back to previous state)
        p1 = value[(0, 0, 1, 2), :] * env # up
        p1[2+1, 2] = value[2+1, 2] # keep the state where next state will be block.
        
        p2 = value[(1, 2, 3, 3), :] * env # down
        p2[2-1, 2] = value[2-1, 2] # keep the state where next state will be block.
        
        p3 = value[:, (0, 0, 1, 2)] * env # left
        p3[2, 2+1] = value[2, 2+1] # keep the state where next state will be block.
        
        p4 = value[:, (1, 2, 3, 3)] * env # right
        p4[2, 2-1] = value[2, 2-1] # keep the state where next state will be block.
        
    value = p0 + 0.25 * (p1+p2+p3+p4)
    print(i+1, value)
