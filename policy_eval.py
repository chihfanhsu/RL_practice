#code
import numpy as np


p = env.get_env_model()
policy = np.ones((4,env.MAZE_Limit[0]*env.MAZE_Limit[1]))*0.25
gamma = 1
reward = np.ones((4,16))*-1
# set reward to oval state = 0
for a in range(len(env.action_space)):
    for t in range(len(env.oval_state)):
        reward[a,env.position2state(env.oval_state[t])] = 0

# set reward to oval state = 0
for a in range(len(env.action_space)):
    for t in range(len(env.block_state)):
        reward[a,env.position2state(env.block_state[t])] = 0

# policy evaluation
v_start = np.zeros((16,))
v = v_start.copy()
for k in range(10):
    v_next = np.sum(policy*(reward+gamma*np.matmul(p,v)),axis = 0)
    print(v_next.reshape([4,4]))
    v = v_next.copy()