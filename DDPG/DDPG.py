# reference from

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(1)
tf.compat.v1.set_random_seed(1)
tf.compat.v1.disable_eager_execution()

class DRL:
    def __init__(self,env):
        self.DDPG = self.Deep_Deterministic_Policy_Gradient(env)
    
    class Deep_Deterministic_Policy_Gradient:
        def __init__(
                self,
                env,
                learning_rate=0.02,
                discount_factor=0.9,
                memory_size=1000,
                batch_size=32,
                output_graph=False
        ):
            self.env = env
            self.n_actions = len(self.env.action_space)
            self.dim_act = 1
            self.n_features = len(self.env.origin)
            self.lr = learning_rate
            self.gamma = discount_factor
            self.memory_size = memory_size
            self.batch_size=batch_size
            self.memory_counter = 0
            self.TAU = 0.01
            self.memory_trans = np.zeros((self.memory_size, self.n_features*2 + 2))
            
            # total learning step
            self.learn_step_counter = 0
            
            # Create sess
            self.sess = tf.compat.v1.Session()
            
            # define inputs
            self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # s
            self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')  # s'
            self.r = tf.compat.v1.placeholder(tf.float32, [None, ], name='r')  # Reward
            
            with tf.compat.v1.variable_scope('Actor'):
                self.a = self._build_actor(self.s, trainable=True, scope='eval')
                self.a_ = self._build_actor(self.s_, trainable=False, scope='target')
            
            with tf.compat.v1.variable_scope('Critic'):
                self.Av = self._build_critic(self.s, self.a, trainable=True, scope='eval')
                self.Av_ = self._build_critic(self.s_, self.a_, trainable=False, scope='target')
            
            # networks parameters
            self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
            self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
            self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
            self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
            
            # target net replacement
            self.soft_replace = [tf.compat.v1.assign(t, (1 - self.TAU)*t + self.TAU*e)
                                 for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
            
            with tf.compat.v1.variable_scope('Critic_loss'):
                self.TD_target = self.r + self.gamma*self.Av_
                self.TD_error = tf.compat.v1.losses.mean_squared_error(labels=self.TD_target, predictions=self.Av)
                
            with tf.compat.v1.variable_scope('Actor_loss'):
                self.actor_error = - tf.reduce_mean(self.Av)
                
            with tf.compat.v1.variable_scope('train'):
                # minimize error
                self.critic_train = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.TD_error, var_list=self.ce_params)
                self.actor_train = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.actor_error, var_list=self.ae_params)
                
            # initial parameter
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
            if output_graph:
                # $ tensorboard --logdir=logs
                tf.summary.FileWriter("logs/", self.sess.graph)
                
        def store_transition(self, s, a, r, s_):
            # print("s:{0}, a:{1}, r:{2}, s_:{3}".format(s, a, r, s_))
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory_trans[index, :] = transition
            self.memory_counter += 1
            
        def choose_action(self, observation):
            action = self.sess.run(self.a, feed_dict={self.s: observation[np.newaxis, :]})
            return np.squeeze(action)
        
        def _build_actor(self, s, trainable, scope):
            with tf.compat.v1.variable_scope(scope):
                f1 = tf.compat.v1.layers.dense(s, 20, activation=tf.nn.relu, name='f1', trainable=trainable)
                A_raw = tf.compat.v1.layers.dense(f1, self.dim_act, activation=tf.math.sigmoid, name='A', trainable=trainable)
                A_scale = tf.multiply(A_raw, (self.n_actions-1), name='scaled_a')
                return A_scale

        def _build_critic(self, s, a, trainable, scope):
            with tf.compat.v1.variable_scope(scope):
                n_l1 = 20
                w1_s = tf.compat.v1.get_variable('w1_s', [self.n_features, n_l1], trainable=trainable)
                w1_a = tf.compat.v1.get_variable('w1_a', [self.dim_act, n_l1], trainable=trainable)
                b1 = tf.compat.v1.get_variable('b1', [self.dim_act, n_l1], trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                Av = tf.compat.v1.layers.dense(net, 1, trainable=trainable)
                return Av
        
        def learn(self):
            self.sess.run(self.soft_replace)
                
            # sample batch memory from all memory
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            
            batch_memory_trans = self.memory_trans[sample_index, :]
            
            self.sess.run(self.actor_train, feed_dict={self.s:  batch_memory_trans[:, :self.n_features]})
            self.sess.run(self.critic_train, feed_dict={self.s:  batch_memory_trans[:, :self.n_features],
                                                        self.a:  batch_memory_trans[:, self.n_features][:,np.newaxis],
                                                        self.r:  batch_memory_trans[:, self.n_features + 1],
                                                        self.s_: batch_memory_trans[:, -self.n_features:]})
        
        def iteration(self, n_episode=10):
            step = 0
            self.hist = []
            act_rand = 0.9
            for episode in range(n_episode):
                # initial observation
                self.env.reset()
                ep_reward= 0
                while True:
                    self.env.render()
                    # RL choose action based on observation
                    observation = self.env.cur_state.copy()
                    
                    if (np.random.uniform(0,1,1) < act_rand):
                        action = np.random.choice(np.arange(0, self.n_actions), p=(np.ones((self.n_actions,))/self.n_actions))
                    else:
                        action = np.rint(self.choose_action(observation)).astype(np.int32)
                    
                    # RL take action and get next observation and reward
                    done, reward = self.env.take_action(action, show_animate = True)
                    observation_ = self.env.cur_state.copy()
                    self.store_transition(observation, action, reward, observation_)
                    
                    if (self.memory_counter > self.memory_size):
                        act_rand = act_rand*0.9995
                        self.learn()
                        
                    ep_reward = ep_reward + reward
                    step += 1
                    if done: # end episode and update model parameter
                        break
                        
                print('Episode:', episode, ' Reward: %i' % int(ep_reward), ' Exploration rate: %.2f' % act_rand)
                self.hist.append([episode, step])
            # output value
            self.Avalue = []
            for s in range(self.env.tot_states):
                coor = self.env.position2state(s, inv = True)
                actions_value_s =  self.critic.Get_Av(coor)
                self.Avalue.append(actions_value_s)
            
            # end of game
            self.env.destroy()
            self.sess.close()