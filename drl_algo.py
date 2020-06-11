# reference from
# https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-DQN1/

import numpy as np
import tensorflow as tf
import model as mdl
from tqdm import tqdm

np.random.seed(1)
tf.compat.v1.set_random_seed(1)
tf.compat.v1.disable_eager_execution()

class DRL:
    def __init__(self,env):
        self.QDN = self.DeepQNetwork(env)
    # Deep Q Network off-policy
    class DeepQNetwork:
        def __init__(
                self,
                env,
                learning_rate=0.01,
                discount_factor=0.9,
                e_greedy=0.5,
                replace_target_iter=200,
                memory_size=2000,
                batch_size=64,
                e_greedy_increment=None,
                output_graph=False
        ):
            self.env = env
            self.n_actions = len(self.env.action_space)
            self.n_features = len(self.env.origin)
            self.lr = learning_rate
            self.gamma = discount_factor
            self.epsilon_max = e_greedy
            self.replace_target_iter = replace_target_iter
            self.memory_size = memory_size
            self.batch_size = batch_size
            self.epsilon_increment = e_greedy_increment
            self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
            
            # total learning step
            self.learn_step_counter = 0
            
            # initialize zero memory [s, a, r, s_]
            self.memory_trans = np.zeros((self.memory_size, self.n_features*2 + 2))
            
            # consist of [target_net, evaluate_net]
            self._build_net()
            
            self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
            
            with tf.compat.v1.variable_scope('hard_replacement'):
                self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
            
            # Create sess
            self.sess = tf.compat.v1.Session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
            if output_graph:
                # $ tensorboard --logdir=logs
                tf.summary.FileWriter("logs/", self.sess.graph)
                
            self.cost_his = []
            
        def _build_net(self):
            # ------------------ build evaluate_net ------------------
            self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
            self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
            self.r = tf.compat.v1.placeholder(tf.float32, [None, ], name='r')  # input Reward
            self.a = tf.compat.v1.placeholder(tf.int32, [None, ], name='a')  # input Action
            
            with tf.compat.v1.variable_scope('eval_net'):
                self.q_eval = mdl.model(self.s,self.n_actions)
                
            with tf.compat.v1.variable_scope('target_net'):
                self.q_next = mdl.model(self.s_,self.n_actions)
                
            with tf.compat.v1.variable_scope('q_target'):
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
                # forbid graident pass back to the network
                self.q_target = tf.stop_gradient(q_target)
                
            with tf.compat.v1.variable_scope('q_eval'):
                a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
                # only output with the certain action
                self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
                
            with tf.compat.v1.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
                
            with tf.compat.v1.variable_scope('train'):
                self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                
        def store_transition(self, s, a, r, s_):
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            # print("s:{0}, a:{1}, r:{2}, s_:{3}".format(s, a, r, s_))
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory_trans[index, :] = transition
            self.memory_counter += 1
            
        def choose_action(self, observation):
            # to have batch dimension when feed into tf placeholder
            if np.random.uniform() < self.epsilon:
                # print('select action')
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval,
                                              feed_dict={self.s: (observation[np.newaxis, :]-self.env.oval_state)/4})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
            return action
        
        def learn(self):
            is_update = False
            # check to replace target parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.replace_target_op)
                is_update = True
                # print('\ntarget_params_replaced\n')
                
            # sample batch memory from all memory
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
                
            batch_memory_trans = self.memory_trans[sample_index, :]
            _, cost = self.sess.run([self._train_op, self.loss],
                                       feed_dict={self.s: (batch_memory_trans[:, :self.n_features]-self.env.oval_state)/4,
                                                  self.s_: (batch_memory_trans[:, -self.n_features:]-self.env.oval_state)/4,  # fixed params
                                                  self.r: batch_memory_trans[:, self.n_features + 1],
                                                  self.a: batch_memory_trans[:, self.n_features]})
            self.cost_his.append(cost)
            
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1
            return is_update
        
        def plot_cost(self):
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(self.cost_his)), self.cost_his)
            plt.ylim(ymax = 5, ymin = 0)
            plt.ylabel('Cost')
            plt.xlabel('training steps')
            plt.show()
            
        def iteration(self, n_episode=10):
            step = 0
            for episode in tqdm(range(n_episode)):
                # initial observation
                self.env.reset()

                while True:
                    self.env.render()
                    # RL choose action based on observation
                    observation = self.env.cur_state.copy()
                    action = self.choose_action(observation)
                    # RL take action and get next observation and reward
                    done, reward = self.env.take_action(action, show_animate = True)
                    observation_ = self.env.cur_state.copy()
                    self.store_transition(observation, action, reward, observation_)
                    if (step > 200)  and (step % 5 == 0):
                        is_update = self.learn()
                        
                    if (episode>(n_episode/2)):
                        self.epsilon = 0.7
                    if  (episode>(n_episode/4)):
                        self.epsilon = 0.9
                        
                    step += 1
                    if done:
                        break
            # output value
            self.Avalue = []
            for s in range(self.env.tot_states):
                coor = self.env.position2state(s, inv = True)
                actions_value_s = self.sess.run(self.q_eval,
                                                feed_dict={self.s: (coor[np.newaxis, :]-self.env.oval_state)/4})
                self.Avalue.append(actions_value_s)
            
            # end of game
            self.env.destroy()
            self.sess.close()