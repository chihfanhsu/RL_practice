# reference from

import numpy as np
import tensorflow as tf
import model as mdl
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(1)
tf.compat.v1.set_random_seed(1)
tf.compat.v1.disable_eager_execution()

class DRL:
    def __init__(self,env):
        self.AC = self.actor_critic(env)
    
    class actor_critic:
        def __init__(
                self,
                env,
                learning_rate=0.001,
                discount_factor=0.9,
                output_graph=False
        ):
            self.env = env
            self.n_actions = len(self.env.action_space)
            self.n_features = len(self.env.origin)
            self.lr = learning_rate
            self.gamma = discount_factor
            
            # total learning step
            self.learn_step_counter = 0
            
            # Create sess
            self.sess = tf.compat.v1.Session()
            self.actor = self.Actor(self.sess, self.lr, self.n_actions, self.n_features)
            self.critic = self.Critic(self.sess, self.gamma, self.lr, self.n_features)
            
            # initial parameter
            self.sess.run(tf.compat.v1.global_variables_initializer())
            
            if output_graph:
                # $ tensorboard --logdir=logs
                tf.summary.FileWriter("logs/", self.sess.graph)
        
         # ------------------ build critic ------------------
        class Critic:
            def __init__(self, sess, discount_factor, learning_rate, n_features):
                self.sess = sess
                self.n_features = n_features
                self.gamma = discount_factor
                self.lr = learning_rate
                self._build_net()
                
            def _build_net(self):
                self.s = tf.compat.v1.placeholder(tf.float32, [1, self.n_features], name='s')  # state
                self.Av_ = tf.compat.v1.placeholder(tf.float32, name='Av_')  # Action value for next state and action Q(s',a')
                self.r = tf.compat.v1.placeholder(tf.float32, name='r')  # Reward

                with tf.compat.v1.variable_scope('critic'): # with parameter w
                    f1 = mdl.fc_blk(self.s, 20, name = 'f1')
                    self.Av = tf.compat.v1.layers.dense(inputs=f1, units=1, activation=None, name="Av")

                    with tf.compat.v1.variable_scope('loss'):
                        # TD: delta = Reward+ gamma*Qw(s',a') - Qw(s,a)
                        self.TD_error = tf.math.reduce_mean(self.r + self.gamma*self.Av_ - self.Av)
                        self.loss =  tf.math.square(self.TD_error)

                    with tf.compat.v1.variable_scope('train'):
                        # minimize error
                        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
             
            def prediction(self, s):
                # calcualte 
                Av = self.sess.run(self.Av, feed_dict={self.s: s[np.newaxis, :]})
                return Av
            
            def train(self, s, r, s_):
                # calcualte Q(a',s')
                Av_ = self.prediction(s_)
                error, _ = self.sess.run([self.TD_error, self.train_op],
                                            feed_dict={self.s: s[np.newaxis, :],
                                                       self.Av_: Av_,
                                                       self.r: r})
                return error
        
        # ------------------ build actor ------------------
        class Actor:
            def __init__(self, sess, learning_rate, n_actions, n_features):
                self.sess = sess
                self.lr = learning_rate
                self.n_actions = n_actions
                self.n_features = n_features
                self._build_net()
                self.cost_his = []
                    
            def _build_net(self):
                self.s = tf.compat.v1.placeholder(tf.float32, [1, self.n_features], name='s')  # state
                self.a = tf.compat.v1.placeholder(tf.int32, name='a')  # action
                self.target = tf.compat.v1.placeholder(tf.float32, name='target')  # error
                
                with tf.compat.v1.variable_scope('actor'): # with parameter theta
                    f1 = mdl.fc_blk(self.s, 20, name = 'f1')
                    act_prob_raw = tf.compat.v1.layers.dense(inputs=f1, units=self.n_actions, activation=None, name="Action")
                    self.act_prob = tf.nn.softmax(act_prob_raw)
                    
                    with tf.compat.v1.variable_scope('loss'):
                        # log pi_theda(s_t,a_t)*error
                        # TD-AC, log pi_theda(s_t,a_t)*TD-error
                        prob_wrt_action = tf.reduce_sum(-tf.math.log(self.act_prob+1e-20)*tf.one_hot(self.a, self.n_actions),axis =1)
                        self.loss = tf.reduce_mean(prob_wrt_action * self.target)

                    with tf.compat.v1.variable_scope('train'):
                        # calculating gamma*grad_log pi_theda(s_t,a_t)*v_t and directly update theda
                        self._train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
            
            def choose_action(self, observation):
                # choose action w.r.t theda
                prob_theda = np.squeeze(self.sess.run(self.act_prob,
                                                      feed_dict={self.s: observation[np.newaxis, :]}))

                action = np.random.choice(range(len(prob_theda)), p=prob_theda)
                return action
       
        
            def train(self, s, a, target):
                _, cost = self.sess.run([self._train_op, self.loss],
                                           feed_dict={self.s: s[np.newaxis, :],
                                                      self.a: a,
                                                      self.target: target})
                self.cost_his.append(cost)
                return cost
            
        def iteration(self, n_episode=10):
            step = 0
            self.hist = []
            act_rand = 0.5
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
                        action = self.actor.choose_action(observation)
                        
                    # RL take action and get next observation and reward
                    done, reward = self.env.take_action(action, show_animate = True)
                    observation_ = self.env.cur_state.copy()
                    error = self.critic.train(observation, reward, observation_)
                    self.actor.train(observation, action, error)
                    act_rand = act_rand*0.9995
                    ep_reward = ep_reward + reward
                    step += 1
                    if done: # end episode and update model parameter
                        break
                        
                print('Episode:', episode, ' Reward: %i' % int(ep_reward), 'Exploration rate: %.2f' % act_rand, )        
                self.hist.append([episode, step])
            # output value
            self.Avalue = []
            for s in range(self.env.tot_states):
                coor = self.env.position2state(s, inv = True)
                actions_value_s =  self.critic.prediction(coor)
                self.Avalue.append(actions_value_s)
            
            # end of game
            self.env.destroy()
            self.sess.close()