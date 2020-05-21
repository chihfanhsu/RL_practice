import os
from maze_env import Maze
from utility import Prediction
import time
import numpy as np

maz = Maze()
# only for interaction:
# env.bind('<KeyPress>', onKeyPress)
# env.mainloop()

s = maz.MAZE_Limit
# crate environment
env = np.tile(1, s)
# terminate state
termin = tuple(map(tuple, np.transpose(maz.oval_state)))
env[termin] = 0
# # block state
# block = tuple(map(tuple, np.transpose(maz.block_state)))
# env[block] = 0


for i in range(10):
    # reward
    r = -1
    p0 = np.tile(r, s) * env
    if i==0:
        # initialize value
        value = np.zeros(s)
        print(i)
        print(value)
    # define next state (if outside of box (wall) -> back to previous state)
    p1 = value[(0, 0, 1, 2), :] * env # up
    p2 = value[(1, 2, 3, 3), :] * env # down
    p3 = value[:, (0, 0, 1, 2)] * env # left
    p4 = value[:, (1, 2, 3, 3)] * env # right

    next_state = np.amax(np.concatenate([p1[np.newaxis], p2[np.newaxis], p3[np.newaxis], p4[np.newaxis]]), 0)
    policy = np.argmax(np.concatenate([p1[np.newaxis], p2[np.newaxis], p3[np.newaxis], p4[np.newaxis]]), 0)
        
    value = p0 + next_state
    print(i+1, policy)
