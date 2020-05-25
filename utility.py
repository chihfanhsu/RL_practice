#code
import numpy as np

class Prediction:
    def __init__(self, env):
        self.DP = self.DP(env)
        self.MFMC = self.model_free_MC(env)
        
    class DP:
        def __init__(self,env):
            self.env = env
            self.init_policy = np.ones((len(self.env.action_space),env.tot_states))/len(self.env.action_space)
            self.discont_factor = 1
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
            gamma = self.discont_factor
            # start iteration
            for k in range(update):
                # policy iteration
                # v_next = self.BEE(self.reward, policy, gamma, p, self.value)
                # policy = self.update_oplicy(p, v_next, mode ="argmax")
                # value iteration
                v_next = self.BOE(self.env.reward, gamma, p, self.value)
                print(k, v_next.reshape([4,4]))
                self.value = v_next.copy()
    
    class model_free_MC:
        def __init__(self,env):
            self.env = env
            self.discont_factor = 1
            self.init_value = np.zeros((self.env.tot_states,))
            self.value = self.init_value.copy()
            # visit matrix N(s)
            self.N = np.zeros((self.env.tot_states,))
            # increment total return S(s)
            self.S = np.zeros((self.env.tot_states,))
            # sample actions
            self.sp = np.ones(shape=(len(self.env.action_space),))
        
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
        
        def iteration(self, n_episode = 10, mode="first_visit"):
            print("The mode is", mode)
            # visit matrix N(s)
            self.N = np.zeros((self.env.tot_states,))
            # increment total return S(s)
            self.S = np.zeros((self.env.tot_states,))
            rm_value = np.zeros((self.env.tot_states,))
            for rp in range(n_episode):
                is_terminate = False
                episode = self.get_episode(self.env, self.sp)
                # get episode
                eN = np.zeros((self.env.tot_states,))
                eS = np.zeros((self.env.tot_states,))
                # scan the eposid
                for step in range(len(episode)):
                    cur_step = np.asarray([episode[step,0],episode[step,1]])
                    nxt_state = np.asarray([episode[step,2],episode[step,3]])

                    # first-visit
                    if mode == "first_visit":
                        if (eN[int(self.env.position2state(cur_step))]==0):
                            eN[int(self.env.position2state(cur_step))] = eN[int(self.env.position2state(cur_step))] + 1
                            Gt = np.sum(episode[step:,4]*np.logspace(start=1,
                                                                     stop=len(episode[step:,4]),
                                                                     num=len(episode[step:,4]),
                                                                     endpoint=True,
                                                                     base=self.discont_factor))
                            eS[int(self.env.position2state(cur_step))] = eS[int(self.env.position2state(cur_step))] + Gt
                    elif mode == "every_visit":
                        eN[int(self.env.position2state(cur_step))] = eN[int(self.env.position2state(cur_step))] + 1
                        Gt = np.sum(episode[step:,4]*np.logspace(start=1,
                                                                 stop=len(episode[step:,4]),
                                                                 num=len(episode[step:,4]),
                                                                 endpoint=True,
                                                                 base=self.discont_factor))
                        eS[int(self.env.position2state(cur_step))] = eS[int(self.env.position2state(cur_step))] + Gt
                    else:
                        print("ERROR MC mode, please set to first_visit or every_visit")

                self.N = self.N + eN
                self.S = self.S + eS
                self.value = np.divide(self.S, self.N, out=np.zeros_like(self.S), where=self.N!=0)
                # running mean
                # rm_value = rm_value + np.divide(eS-rm_value, N, out=np.zeros_like(eS), where=eN!=0)
            
            print("N\n",self.N.reshape(4,4))
            # print("Runing V\n",rm_value.reshape(4,4))
            print("V\n",self.value.reshape(4,4))
            print("--------------------------\n")