#code
import numpy as np

class Prediction:
    def __init__(self, env):
        super(Prediction, self).__init__()
        self.env = env
        self.tot_states = self.env.MAZE_Limit[0]*self.env.MAZE_Limit[1]
        self.init_policy = np.ones((len(self.env.action_space),self.tot_states))/len(self.env.action_space)
        self.reward = np.ones((len(self.env.action_space),self.tot_states))*-1
        self.discont_factor = 1
        self.init_value = np.zeros((self.tot_states,))
        self.value = self.init_value.copy()
        # set reward to oval state = 0
        for a in range(len(self.env.action_space)):
            for t in range(len(self.env.oval_state)):
                self.reward[a,self.env.position2state(self.env.oval_state[t])] = 0
        
        # set reward to oval state = 0
        for a in range(len(self.env.action_space)):
            for t in range(len(self.env.block_state)):
                self.reward[a,self.env.position2state(self.env.block_state[t])] = 0
        
    def update_oplicy(self, p, value, mode = "argmax"):
        if mode == "argmax":
            # Greedy Policy Improvement
            new_policy = np.zeros((len(self.env.action_space),16))
            argmax_a = np.argmax(np.matmul(p, value),axis=0)
            for c, value in enumerate(argmax_a):
                new_policy[value,c] = 1
            return new_policy
        else:
            print("Error parameter (model) in update_oplicy function.")
            return False
        
    def BEE(self, reward, policy, gamma, p, v): # Bellman Expectation Equation
            return np.sum(policy*(reward+gamma*np.matmul(p,v)),axis = 0)
    
    def BOE(self, reward, gamma, p, v): # Bellman Optimality Equation
        return np.max(reward + gamma*np.matmul(p,v), axis = 0)
    
    def iteration(self, update=10):
        p = self.env.get_env_model()
        policy = self.init_policy.copy()
        self.value = self.init_value.copy()
        gamma = self.discont_factor
        # policy evaluation
        for k in range(update):
            # policy iteration
            # v_next = self.BEE(self.reward, policy, gamma, p, self.value)
            # policy = self.update_oplicy(p, v_next, mode ="argmax")
            # value iteration
            v_next = self.BOE(self.reward, gamma, p, self.value)
            print(k, v_next.reshape([4,4]))
            self.value = v_next.copy()
