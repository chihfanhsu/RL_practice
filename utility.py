#code
import numpy as np

class Prediction:
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
                # v_next = self.BEE(self.reward, policy, gamma, p, self.value)
                # policy = self.update_oplicy(p, v_next, mode ="argmax")
                # value iteration
                v_next = self.BOE(self.env.reward, self.gamma, p, self.value)
                self.value = v_next.copy()
                
            print(k, self.value.reshape([4,4]))
            
    class model_free:
        def __init__(self, env, lamb = 0.9, discount_factor = 0.9, alpha = 0.5):
            self.env = env
            self.gamma = discount_factor
            self.lamb = lamb
            self.init_value = np.zeros((self.env.tot_states,))
            self.value = self.init_value.copy()
            # visit matrix N(s)
            self.N = np.zeros((self.env.tot_states,))
            # increment total return S(s)
            self.S = np.zeros((self.env.tot_states,))
            # sample actions
            self.sp = np.ones(shape=(len(self.env.action_space),))
            self.alpha = alpha
        
        def get_episode(self, env, sp):
            episode = []
            while (1):
                cur_state = self.env.cur_state.copy()
                is_terminate, reward = self.env.take_action(np.random.choice(np.arange(0, len(self.env.action_space)), p=sp/np.sum(sp)), show_animate=False)
                nxt_state = self.env.cur_state.copy()
                episode.append(np.array([cur_state[0],cur_state[1],nxt_state[0],nxt_state[1],reward]))
                if is_terminate:
                    self.env.cur_state = self.env.start_state.copy()
                    episode = np.asarray(episode)
                    break
            return episode
        
        def iteration(self, n_episode = 10, model="MC"):
            print("The model is", model)
            # visit matrix N(s)
            self.N = np.zeros((self.env.tot_states,))
            # increment total return S(s)
            self.S = np.zeros((self.env.tot_states,))
            rm_value = np.zeros((self.env.tot_states,))
            for rp in range(n_episode):
                is_terminate = False
                episode = self.get_episode(self.env, self.sp)
                if (model == "MC"):
                    # get episode
                    eN = np.zeros((self.env.tot_states,))
                    eS = np.zeros((self.env.tot_states,))
                    # scan the eposid
                    for step in range(len(episode)):
                        cur_state = int(self.env.position2state(np.asarray([episode[step,0],episode[step,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([episode[step,2],episode[step,3]])))
                        # first-visit
                        if (eN[cur_state]==0):
                            eN[cur_state] = eN[cur_state] + 1
                            Gt = np.sum(episode[step:,4]*np.logspace(start=1,
                                                                     stop=len(episode[step:,4]),
                                                                     num=len(episode[step:,4]),
                                                                     endpoint=True,
                                                                     base=self.gamma))
                            eS[cur_state] = eS[cur_state] + Gt
                        # every_visit"
                        # eN[int(self.env.position2state(cur_step))] = eN[int(self.env.position2state(cur_step))] + 1
                        # Gt = np.sum(episode[step:,4]*np.logspace(start=1,
                        #                                          stop=len(episode[step:,4]),
                        #                                          num=len(episode[step:,4]),
                        #                                          endpoint=True,
                        #                                          base=self.gamma))
                        # eS[int(self.env.position2state(cur_step))] = eS[int(self.env.position2state(cur_step))] + Gt

                    self.N = self.N + eN
                    self.S = self.S + eS
                    # regular mean
                    # self.value = np.divide(self.S, self.N, out=np.zeros_like(self.S), where=self.N!=0)
                    # running mean
                    # V(s) = V(s) + (Gt-V(s))/N(s), s represent s_t
                    self.value = self.value + np.divide(eS-self.value, self.N, out=np.zeros_like(eS), where=eN!=0)
                    
                elif (model == "TD_0"):
                    E = np.zeros((self.env.tot_states,))
                    for step in range(len(episode)):
                        cur_state = int(self.env.position2state(np.asarray([episode[step,0],episode[step,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([episode[step,2],episode[step,3]])))
                        reward = episode[step,4]
                        TD_target = reward + self.gamma*self.value[nxt_state]
                        TD_error = TD_target - self.value[cur_state]
                        simulas = np.zeros((self.env.tot_states,))
                        simulas[cur_state] = 1
                        # update Eligibility Traces 
                        E = simulas
                        # update value
                        self.value = self.value + self.alpha*TD_error*E
                elif (model == "TD_n"):
                    n = 5
                    for step in range(len(episode)):
                        sub_episode = episode[step:(step+5)]
                        cur_state = int(self.env.position2state(np.asarray([sub_episode[0,0],sub_episode[0,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([sub_episode[len(sub_episode)-1,2],sub_episode[len(sub_episode)-1,3]])))
                        reward = sub_episode[:,4]
                        # print(sub_episode)
                        # print(reward)
                        TD_target = np.sum(reward*np.logspace(start=1,
                                                              stop=len(reward),
                                                              num=len(reward),
                                                              endpoint=True,
                                                              base=self.gamma)) + np.power(self.gamma,len(reward))*self.value[nxt_state]
                        
                        TD_error = TD_target - self.value[cur_state]
                        self.value[cur_state] = self.value[cur_state] + self.alpha*(TD_error)
                elif (model == "TD_lambda"):
                    # backward
                    # Eligibility Traces
                    E = np.zeros((self.env.tot_states,))
                    for step in range(len(episode)):
                        cur_state = int(self.env.position2state(np.asarray([episode[step,0],episode[step,1]])))
                        nxt_state = int(self.env.position2state(np.asarray([episode[step,2],episode[step,3]])))
                        reward = episode[step,4]
                        TD_target = reward + self.gamma*self.value[nxt_state]
                        TD_error = TD_target - self.value[cur_state]
                        simulas = np.zeros((self.env.tot_states,))
                        simulas[cur_state] = 1
                        # update Eligibility Traces 
                        E = self.lamb*self.gamma*E + simulas
                        # update value
                        self.value = self.value + self.alpha*TD_error*E
                else:
                    print("ERROR MF model, please set to MC or TD")
                    
#             print("N\n",self.N.reshape(4,4))
            # print("Runing V\n",rm_value.reshape(4,4))
            print("V\n",self.value.reshape(4,4))
            print("--------------------------\n")