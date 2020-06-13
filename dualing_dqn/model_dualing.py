import tensorflow as tf

def fc_blk(inputs, nodes, name = 'fc_blk'):
    with tf.compat.v1.variable_scope(name) as scope:
        fc = tf.compat.v1.layers.dense(inputs=inputs, units=nodes, activation=None, name="fc")
        #bn = batch_norm(fc, phase_train)
        act = tf.nn.relu(fc, name= "act")
        return act
    
def FA(state, n_actions):
    with tf.compat.v1.variable_scope('agent'):
        f1 = fc_blk(state, 20, name = 'f1')
        values = tf.compat.v1.layers.dense(inputs=f1, units=n_actions, activation=None, name="values")
        with tf.compat.v1.variable_scope('agent_value'):
            v1 = fc_blk(f1, 10, name = 'v1')
            V = fc_blk(v1, 1, name = 'V')
        with tf.compat.v1.variable_scope('agent_advantage'):
            a1 = fc_blk(f1, 20, name = 'a1')
            A = fc_blk(a1, n_actions, name = 'A')
        value = V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))
        return values

def model(observ, n_actions):
    with tf.compat.v1.variable_scope('mdl'):
        values = FA(observ,n_actions)
        return values