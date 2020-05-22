import os
from maze_env import Maze
from utility import Prediction
import time
import numpy as np

maz = Maze()
# only for interaction:
# env.bind('<KeyPress>', onKeyPress)
# env.mainloop()
# whether there's block or not
block_setting = True
# choose which algorithm type
# [pe]: policy evaluation
# [vi]: value(policy) iteration 
algo_type = "vi"

# initiate matrix with dimension from maze
s = maz.MAZE_Limit
# crate environment
env = np.tile(1, s)
# terminate state
termin = tuple(map(tuple, np.transpose(maz.oval_state)))
env[termin] = 0
# block state
if block_setting:
    block0 = maz.block_state
    block = tuple(map(tuple, np.transpose(block0)))
    env[block] = 0

# boundary action
# right [1, 2, 3, 3]
rt = list(range(s[1])) + np.ones(s[1], dtype=int)
rt[np.greater(rt, s[1] - 1)] = s[1] - 1
# left [0, 0, 1, 2]
lt = list(range(s[1])) - np.ones(s[1], dtype=int)
lt[np.less(lt,0)] = 0
# up  [0, 0, 1, 2]
ut = list(range(s[0])) - np.ones(s[0], dtype=int)
ut[np.less(ut,0)] = 0
# down [1, 2, 3, 3]
dt = list(range(s[0])) + np.ones(s[0], dtype=int)
dt[np.greater(dt, s[0] - 1)] = s[0] - 1

if block_setting:
    # block action
    # right (from left)
    rt_block = block0.copy()
    b1 = rt_block[..., 1] - 1
    b1[np.less(b1, 0)] = 0
    rt_block[..., 1] = b1
    # left (from right)
    lt_block = block0.copy()
    b2 = lt_block[..., 1] + 1
    b2[np.greater(b2, s[1] - 1)] = s[1] - 1
    lt_block[..., 1] = b2
    # up (from down)
    ut_block = block0.copy()
    b3 = ut_block[..., 0] + 1
    b3[np.greater(b3, s[0] - 1)] = s[0] - 1
    ut_block[..., 0] = b3
    # down (from up)
    dt_block = block0.copy()
    b4 = dt_block[..., 0] - 1 
    b4[np.less(b4, 0)] = 0
    dt_block[..., 0] = b4

# reward
r0 = -1
reward = np.tile(r0, s) * env
for i in range(10):
    if i==0:
        # initialize value
        value = np.zeros(s, dtype=int)
        print(i)
        print(value)
    # define next state (if outside of box (wall) -> back to previous state)
    # action
    p1 = value[:, rt]  # right
    p2 = value[:, lt] # left
    p3 = value[ut, :] # up
    p4 = value[dt, :] # down
    # block action
    if block_setting:
        p1[rt_block] = value[rt_block] # right (from left)
        p2[lt_block] = value[lt_block] # left (from right)
        p3[ut_block] = value[ut_block] # up (from down)
        p4[dt_block] = value[dt_block] # down (from up)
    
    p1 = p1 * env
    p2 = p2 * env
    p3 = p3 * env
    p4 = p4 * env
    
    if (algo_type == "pe"): # policy evaluation
        p0 = 0.25
        next_state = p0 * (p1+p2+p3+p4)
    elif (algo_type == "vi"): # value(policy) iteration
        policy = np.argmax(np.concatenate([p1[np.newaxis], p2[np.newaxis], p3[np.newaxis], p4[np.newaxis]]), 0)
        next_state = np.amax(np.concatenate([p1[np.newaxis], p2[np.newaxis], p3[np.newaxis], p4[np.newaxis]]), 0)
    else:
        print("type error")
    
    value = reward + next_state
    print(i+1, value)
