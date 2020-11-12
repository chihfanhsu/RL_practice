# Modify from
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.1_Double_DQN/RL_brain.py

import numpy as np
import tensorflow as tf
import model as mdl
from tqdm import tqdm

np.random.seed(1)
tf.compat.v1.set_random_seed(1)
tf.compat.v1.disable_eager_execution()

class DRL:
    def __init__(self, env, double_q, sess):
        self.QDN = self.DeepQNetwork(env, double_q = double_q, sess = sess)
    # Deep Q Network off-policy
    class DeepQNetwork:
        def __init__(
                self,
                env,
                double_q = True,
                sess = None,
                e_greedy_increment=None,
                output_graph=False,
                learning_rate=0.01,
                discount_factor=0.9,
                e_greedy=0.5,
                replace_target_iter=200,
                memory_size=2000,
                batch_size=64,
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
            self.double_q =double_q
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
            
            if sess is None:
                # Create sess
                self.sess = tf.compat.v1.Session()
                self.sess.run(tf.compat.v1.global_variables_initializer())
            else:
                self.sess = sess
            
            if output_graph:
                # $ tensorboard --logdir=logs
                tf.summary.FileWriter("logs/", self.sess.graph)
                
            self.cost_his = []
            
        def _build_net(self):
            # ------------------ build evaluate net ------------------
            self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input
            self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions], name='Q_target')
            
            with tf.compat.v1.variable_scope('eval_net'): # with parameter theta
                self.q_eval = mdl.model(self.s, self.n_actions) # double DQN
            
            with tf.compat.v1.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval, name='TD_error'))
            
            with tf.compat.v1.variable_scope('train'):
                self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
            # ------------------ build target net ------------------
            self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
            
            with tf.compat.v1.variable_scope('target_net'): # with parameter theta'
                self.q_next = mdl.model(self.s_,self.n_actions)
                
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
            observation = observation[np.newaxis, :]
            if np.random.uniform() < self.epsilon:
                # print('select action')
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(actions_value)
                # running Q if nessary
                if not hasattr(self, 'q'):  # record action value it gets
                    self.q = []                    
                    self.running_q = 0
                    
                self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
                self.q.append(self.running_q)
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
            
            # calculate loss = r + gamma*Q_t(s', argmax_a' Q_e(s',a')) - Q_e(s,a)
            # estimate Q_t(s',a') and Q_e(s',a')
            q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
                                                feed_dict={self.s:  batch_memory_trans[:, -self.n_features:],
                                                           self.s_: batch_memory_trans[:, -self.n_features:]})
            # estimate Q_e(s,a)
            q_eval = self.sess.run(self.q_eval, {self.s: batch_memory_trans[:, :self.n_features]}) # Q_e(s,a)
            
            q_target = q_eval.copy()
            
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            # actual action a
            eval_act_index = batch_memory_trans[:, self.n_features].astype(int)
            # reward
            reward = batch_memory_trans[:, self.n_features + 1]
            
            if self.double_q:
                max_act4next = np.argmax(q_eval4next, axis=1) # argmax_a' Q_e(s',a')
                selected_q_next = q_next[batch_index, max_act4next] # Q_t(s', argmax_a' Q_e(s',a'))
            else:
                selected_q_next = np.max(q_next, axis=1) # Q_t(s', a')
            
            # modify the tesnor Q_e(s,a) to r + gamma*Q_t(s', argmax_a' Q_e(s',a')) with respect to actual action a
            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
            
            # update network by loss calculate loss = r + gamma*Q_t(s', argmax_a' Q_e(s',a')) - Q_e(s,a)
            _, cost = self.sess.run([self._train_op, self.loss],
                                       feed_dict={self.s: batch_memory_trans[:, :self.n_features],
                                                  self.q_target: q_target})
            
            self.cost_his.append(cost)
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1
            return is_update
            
        def iteration(self, n_episode=10):
            step = 0
            self.hist = []
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
                    if (step > 0)  and (step % 5 == 0):
                        is_update = self.learn()
                        
                    if (episode>(n_episode/2)):
                        self.epsilon = 0.7
                    if  (episode>(3*n_episode/4)):
                        self.epsilon = 0.9
                        
                    step += 1
                    if done:
                        break
                
                self.hist.append([episode, step])
            # output value
            self.Avalue = []
            for s in range(self.env.tot_states):
                coor = self.env.position2state(s, inv = True)
                actions_value_s = self.sess.run(self.q_eval,
                                                feed_dict={self.s: (coor[np.newaxis, :]),
                                                           self.s_: (coor[np.newaxis, :])})
                self.Avalue.append(actions_value_s)