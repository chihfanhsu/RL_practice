# reference from
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py

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
        self.PG = self.policy_gradient(env)
    
    class policy_gradient:
        def __init__(
                self,
                env,
                learning_rate=0.01,
                discount_factor=0.9,
                output_graph=False
        ):
            self.env = env
            self.n_actions = len(self.env.action_space)
            self.n_features = len(self.env.origin)
            self.lr = learning_rate
            self.gamma = discount_factor
            self.states = []
            self.actions = []
            self.rewards = []
            
            # total learning step
            self.learn_step_counter = 0
            
            # consist of [target_net, evaluate_net]
            self._build_net()
            
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
            self.v_t = tf.compat.v1.placeholder(tf.float32, [None, ], name='vt')  # input Reward
            self.a = tf.compat.v1.placeholder(tf.int32, [None, ], name='a')  # input Action
            
            with tf.compat.v1.variable_scope('model'): # with parameter theta'
                act_prob_raw = mdl.model(self.s, self.n_actions)
                self.act_prob = tf.nn.softmax(act_prob_raw)
                
            with tf.compat.v1.variable_scope('loss'):
                # log pi_theda(s_t,a_t)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_prob_raw, labels=self.a) # this is negative log of chosen action
                #log_prob = tf.reduce_sum(-tf.math.log(self.act_prob)*tf.one_hot(self.a, self.n_actions))
                # log pi_theda(s_t,a_t)*v_t
                self.loss = tf.reduce_mean(log_prob*self.v_t)
                
            with tf.compat.v1.variable_scope('train'):
                # calculating gamma*grad_log pi_theda(s_t,a_t)*v_t and directly update theda
                self._train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)
                
        def store_transition(self, s, a, r, s_):
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            
        def choose_action(self, observation):
            # choose action w.r.t theda
            prob_theda = np.squeeze(self.sess.run(self.act_prob,
                                                  feed_dict={self.s: observation[np.newaxis, :]}))
            
            action = np.random.choice(range(len(prob_theda)), p=prob_theda)
            return action
        
        def discount_and_norm_rewards(self, rewards):
            # discount episode rewards
            discounted_rewards = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(0, len(rewards))):
                running_add = running_add * self.gamma + self.rewards[t]
                discounted_rewards[t] = running_add

            # normalize episode rewards (tricky to process reward)
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards)
            return discounted_rewards
        
        def learn(self):
            # discount and normalize episode reward
            discounted_norm_rewards = self.discount_and_norm_rewards(self.rewards)
            
            _, cost = self.sess.run([self._train_op, self.loss],
                                       feed_dict={self.s: np.vstack(self.states),
                                                  self.v_t: discounted_norm_rewards,
                                                  self.a: np.array(self.actions)})
            self.cost_his.append(cost)
            self.learn_step_counter += 1
            return discounted_norm_rewards
            
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
                    step += 1
                    if done: # end episode and update model parameter
                        reward_sum = sum(self.rewards)
                        print("episode:", episode, "reward:", int(reward_sum))
                            
                        vt = self.learn()
                        # empty episode
                        self.states = []
                        self.actions = []
                        self.rewards = []
                        break
                self.hist.append([episode, step])
            
            # end of game
            self.env.destroy()
            self.sess.close()