#code
import numpy as np

class Prediction:
    def __init__(self, env):
        super(Prediction, self).__init__()
        self.env = env
        
    def policy_eval(self, iteration=10):
        p = self.env.get_env_model()
        policy = np.ones((4,self.env.MAZE_Limit[0]*self.env.MAZE_Limit[1]))*0.25
        gamma = 1
        reward = np.ones((4,16))*-1
        # set reward to oval state = 0
        for a in range(len(self.env.action_space)):
            for t in range(len(self.env.oval_state)):
                reward[a,self.env.position2state(self.env.oval_state[t])] = 0

        # set reward to oval state = 0
        for a in range(len(self.env.action_space)):
            for t in range(len(self.env.block_state)):
                reward[a,self.env.position2state(self.env.block_state[t])] = 0

        # policy evaluation
        v_start = np.zeros((16,))
        v = v_start.copy()
        for k in range(iteration):
            v_next = np.sum(policy*(reward+gamma*np.matmul(p,v)),axis = 0)
            print(k, v_next.reshape([4,4]))
            v = v_next.copy()