#code
import numpy as np

class RL:
    def __init__(self, env):
        self.DP = self.DP(env)
        self.MF = self.model_free(env)
        
    class DP:
        def __init__(self,env,discount_factor=1):
            self.env = env
            self.init_policy = np.ones((len(self.env.action_space),env.tot_states))/len(self.env.action_space)
            self.gamma = discount_factor
            self.init_value = np.zeros((env.tot_states,))
            self.value = self.init_value.copy()
        
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

        # Bellman Expectation Equation
        def BEE(self, reward, policy, gamma, p, v):
                return np.sum(policy*(reward+gamma*np.matmul(p,v)),axis = 0)

        # Bellman Optimality Equation
        def BOE(self, reward, gamma, p, v):
            return np.max(reward + gamma*np.matmul(p,v), axis = 0)

        def iteration(self, update=10):
            p = self.env.get_env_model()
            policy = self.init_policy.copy()
            self.value = self.init_value.copy()
            # start iteration
            for k in range(update):
                # policy iteration
                v_next = self.BEE(self.env.reward, policy, self.gamma, p, self.value)
                # policy = self.update_oplicy(p, v_next, mode ="argmax")
                # value iteration
                # v_next = self.BOE(self.env.reward, self.gamma, p, self.value)
                self.value = v_next.copy()
                
            print(k, self.value.reshape([4,4]))
            
    class model_free:
        def __init__(self, env, lamb=0.9, discount_factor=0.9, alpha=0.5, epsilon=0.5):
            self.env = env
            self.gamma = discount_factor
            self.lamb = lamb
            self.alpha = alpha
            self.epsilon = epsilon
            # init value matrix
            self.init_value = np.zeros((self.env.tot_states,)) # state-value
            self.value = self.init_value.copy() # state-value            
            self.init_Avalue = np.zeros((len(self.env.action_space),self.env.tot_states)) # action-value
            self.Avalue = self.init_Avalue.copy() # action-value
            # sample actions
            self.init_policy = np.ones((len(self.env.action_space),self.env.tot_states))/len(self.env.action_space)
            self.policy = self.init_policy.copy()
        
        def get_episode(self, env, policy):
            episode = []
            while (1):
                cur_state = self.env.cur_state.copy()
                state_policy = policy[:,int(self.env.position2state(cur_state))]
                # take action
                action = np.random.choice(np.arange(0, len(self.env.action_space)), p=state_policy)
                is_terminate, reward = self.env.take_action(action, show_animate=False)
                nxt_state = self.env.cur_state.copy()
                # save to episode
                episode.append(np.array([cur_state[0],cur_state[1],nxt_state[0],nxt_state[1], reward, action]))
                if is_terminate:
                    self.env.cur_state = self.env.start_state.copy()
                    episode = np.asarray(episode)
                    break
            return episode
        
        def epsilon_greedy(self, Avalue, epsilon=0.5):
            new_policy = np.ones((len(self.env.action_space), self.env.tot_states))*epsilon/len(self.env.action_space)
            argmax_a = np.argmax(Avalue, axis=0)
            for c, value in enumerate(argmax_a):
                new_policy[value,c] = epsilon/len(self.env.action_space) + 1 - epsilon
                
            return new_policy
        
        def iteration(self, n_episode = 10, model="MC", control = False, on_policy = True):
            print("The model is", model)
            # visit matrix N(s)
            sN = np.zeros((self.env.tot_states,)) # state-value
            N = np.zeros((len(self.env.action_space),self.env.tot_states)) # action-value
            # increment total return S(s)
            sS = np.zeros((self.env.tot_states,)) # state-value
            S = np.zeros((len(self.env.action_space),self.env.tot_states)) # action-value
            self.value = self.init_value.copy() # state-value
            self.Avalue = self.init_Avalue.copy() # action-value
            self.policy = self.init_policy.copy()
            # learning from episodes
            for rp in range(n_episode):
                # get an episode sequence
                episode = self.get_episode(self.env, self.policy)
                if (model == "MC"):
                    # initial counting and return matrices for a episode
                    eN = np.zeros((len(self.env.action_space),self.env.tot_states)) # action-value
                    eS = np.zeros((len(self.env.action_space),self.env.tot_states)) # action-value
                    esN = np.zeros((self.env.tot_states,)) # state-value
                    esS = np.zeros((self.env.tot_states,)) # state-value
                    # scan the eposid
                    for t in range(len(episode)):
                        # get information from the state s at time t
                        cur_state = int(self.env.position2state(np.asarray([episode[t,0],episode[t,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([episode[t,2],episode[t,3]])))
                        action = int(episode[t,5])
                        # first-visit method. for every_visit method, remove the if (eN[cur_state]==0)
                        # for action-value
                        if (eN[action, cur_state]==0):
                            eN[action, cur_state] = eN[action, cur_state] + 1
                            # calculate with following returns
                            Gt = np.sum(episode[t:,4]*np.array([self.gamma**i for i in range(len(episode[t:,4]))]))
                            eS[action, cur_state] = eS[action, cur_state] + Gt
                        # for state-value
                        if (esN[cur_state]==0):
                            esN[cur_state] = esN[cur_state] + 1
                            # calculate with following returns
                            Gt = np.sum(episode[t:,4]*np.array([self.gamma**i for i in range(len(episode[t:,4]))]))
                            esS[cur_state] = esS[cur_state] + Gt
                    
                    N = N + eN
                    S = S + eS
                    sN = sN + esN
                    sS = sS + esS
                    # regular mean, S/N
                    # self.value = np.divide(S, self.N, out=np.zeros_like(S), where=N!=0)
                    # running mean, V(s) = V(s) + (Gt-V(s))/N(s)
                    self.Avalue = self.Avalue + np.divide(eS-self.Avalue, N, out=np.zeros_like(eS), where=eN!=0)
                    self.value = self.value + np.divide(esS-self.value, sN, out=np.zeros_like(esS), where=esN!=0)
                    if (control):
                        # update policy
                        self.policy = self.epsilon_greedy(self.Avalue)
                     
                elif (model == "TD_0"):
                    # the codes are similar with TD_lambda except E = simulas
                    # init Eligibility Traces
                    E = np.zeros((len(self.env.action_space),self.env.tot_states))
                    # scan episode
                    for t in range(len(episode)):
                        cur_state = int(self.env.position2state(np.asarray([episode[t,0],episode[t,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([episode[t,2],episode[t,3]])))
                        reward = episode[t,4]
                        action = int(episode[t,5])
                        if (t+1) < len(episode):
                            nxt_action = int(episode[t+1,5])
                        elif (t+1) == len(episode):
                            nxt_action = np.random.choice(np.arange(0, len(self.env.action_space)), p=self.policy[:,nxt_state])
                        else:
                            print("Error")
                        
                        TD_target = reward + self.gamma*self.Avalue[nxt_action, nxt_state]
                        TD_error = TD_target - self.Avalue[action, cur_state]
                        # set 1 for the visiting state
                        simulas = np.zeros((len(self.env.action_space),self.env.tot_states))
                        simulas[action, cur_state] = 1
                        E = simulas
                        # update value
                        self.Avalue = self.Avalue + self.alpha*TD_error*E
                        
                elif (model == "TD_n"):
                    # set length
                    n = 5
                    # scan episode
                    for t in range(len(episode)):
                        # get partial episode at most 5 states
                        sub_episode = episode[t:(t+5)]
                        cur_state = int(self.env.position2state(np.asarray([sub_episode[0,0],sub_episode[0,1]])))
                        # get the nxt_state at the end of the sub_episode
                        nxt_state = int(self.env.position2state(np.asarray([sub_episode[len(sub_episode)-1,2],sub_episode[len(sub_episode)-1,3]])))
                        # get reward sequence
                        reward = sub_episode[:,4]
                        action = int(episode[t,5])
                        if (t+1) < len(episode):
                            nxt_action = int(episode[t+1,5])
                        elif (t+1) == len(episode):
                            nxt_action = np.random.choice(np.arange(0, len(self.env.action_space)), p=self.policy[:,nxt_state])
                        else:
                            print("Error")
                        # calculate the TD target, R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n} + gamma^{n}*V_{t+n} 
                        TD_target = np.sum(reward*np.array([self.gamma**i for i in range(len(reward))])) + np.power(self.gamma,len(reward))*self.Avalue[nxt_action, nxt_state]
                        
                        TD_error = TD_target - self.Avalue[action, cur_state]
                        self.Avalue[action, cur_state] = self.Avalue[action, cur_state] + self.alpha*(TD_error)
                        
                elif (model == "TD_lambda"): # Sarsa_lambda
                    # backward
                    # Eligibility Traces
                    E = np.zeros((len(self.env.action_space),self.env.tot_states))
                    # scan episode
                    for t in range(len(episode)):
                        cur_state = int(self.env.position2state(np.asarray([episode[t,0],episode[t,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([episode[t,2],episode[t,3]])))
                        reward = episode[t,4]
                        action = int(episode[t,5])
                        if (t+1) < len(episode):
                            if on_policy:
                                nxt_action = int(episode[t+1,5]) # on policy
                            else:
                                nxt_action = np.argmax(self.Avalue[:,nxt_state]) # off policy
                        elif (t+1) == len(episode):
                            nxt_action = np.random.choice(np.arange(0, len(self.env.action_space)), p=self.policy[:,nxt_state])
                        else:
                            print("Error")
                        
                        TD_target = reward + self.gamma*self.Avalue[nxt_action, nxt_state]
                        TD_error = TD_target - self.Avalue[action, cur_state]
                        # set 1 for the visiting state
                        simulas = np.zeros((len(self.env.action_space),self.env.tot_states))
                        simulas[action, cur_state] = 1
                        E = E + simulas
                        # update V(s) for every state
                        self.Avalue = self.Avalue + self.alpha*TD_error*E
                        # update Eligibility Traces
                        E = self.gamma*self.lamb*E
                    if (control):
                        # update policy
                        self.policy = self.epsilon_greedy(self.Avalue)
                    
                else:
                    print("ERROR MF model")
