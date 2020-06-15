import tensorflow as tf

def fc_blk(inputs, nodes, trainable, name = 'fc_blk'):
    with tf.compat.v1.variable_scope(name) as scope:
        fc = tf.compat.v1.layers.dense(inputs=inputs, units=nodes, activation=None, trainable=trainable, name="fc")
        #bn = batch_norm(fc, phase_train)
        act = tf.nn.relu(fc, name= "act")
        return act