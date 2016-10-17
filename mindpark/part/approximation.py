import tensorflow as tf
from mindpark.model.layer import dense, conv2d, rnn


def q_function(model, network, observs, actions, config=None):
    """
    Action value approximation.
    """
    with tf.variable_scope('behavior'):
        state = model.add_input('state', observs)
        hidden = network(model, state)
        qvalues = dense(hidden, actions, tf.identity)
        qvalues = model.add_output('qvalues', qvalues)
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        model.add_output('value', tf.reduce_max(qvalues, 1))
        model.add_cost(
            'cost', (tf.reduce_sum(action * qvalues, 1) - return_) ** 2)


def policy_gradient(model, network, observs, actions, config):
    """
    Policy gradient of the advantage function. Estimates the advantage from a
    learned value function and experiences returns.
    """
    with tf.variable_scope('behavior'):
        state = model.add_input('state', observs)
        hidden = network(model, state)
        value = model.add_output(
            'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(value, actions, tf.nn.softmax)
        model.add_output(
            'choice', tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
        model.add_cost('cost', return_ * logprob + config.regularize * entropy)


def advantage_policy_gradient(model, network, observs, actions, config):
    """
    Policy gradient of the advantage function. Estimates the advantage from a
    learned value function and experiences returns.
    """
    with tf.variable_scope('behavior'):
        state = model.add_input('state', observs)
        hidden = network(model, state)
        value = model.add_output(
            'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(value, actions, tf.nn.softmax)
        model.add_output(
            'choice', tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
        advantage = tf.stop_gradient(return_ - value)
        actor = advantage * logprob + config.regularize * entropy
        critic = config.scale_critic_loss * (return_ - value) ** 2 / 2
        model.add_cost('cost', critic - actor)


def approx_advantage_policy_gradient(model, network, observs, actions, config):
    """
    Policy gradient of the advantage function. Estimates the advantage from
    learned value and action-value functions.
    """
    with tf.variable_scope('behavior'):
        state = model.add_input('state', observs)
        hidden = network(model, state)
        value = model.add_output(
            'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        advantages = model.add_output(
            'advantages', dense(hidden, actions, tf.identity))
        policy = dense(value, actions, tf.nn.softmax)
        model.add_output(
            'choice', tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        return_ = model.add_input('return_')
        action = tf.one_hot(action, actions)
        with tf.variable_scope('value'):
            critic_v = (return_ - value) ** 2 / 2
        with tf.variable_scope('advantage'):
            advantage = tf.reduce_max(action * advantages, [1])
            qvalue = value + advantage
            critic_q = (return_ - qvalue) ** 2 / 2
        with tf.variable_scope('policy'):
            advantage = tf.stop_gradient(advantage)
            logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
            entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
            actor = advantage * logprob + config.regularize * entropy
        critic = config.scale_critic_loss * (critic_v + critic_q)
        model.add_cost('cost', critic - actor)
