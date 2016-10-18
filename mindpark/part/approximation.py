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
        model.add_output('choice', tf.argmax(qvalues, 1))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        model.add_output('qvalue', tf.reduce_max(qvalues, 1))
        model.add_cost(
            'cost', (tf.reduce_sum(action * qvalues, 1) - return_) ** 2)


def policy_gradient(model, network, observs, actions, config):
    """
    Policy gradient of the return.
    """
    with tf.variable_scope('behavior'):
        state = model.add_input('state', observs)
        hidden = network(model, state)
        value = model.add_output(
            'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
        policy = dense(value, actions, tf.nn.softmax)
        model.add_output('choice', tf.squeeze(tf.multinomial(policy, 1), [1]))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
        actor = config.actor_weight * return_ * logprob
        entropy = config.entropy_weight * entropy
        model.add_cost('cost', -actor + -entropy)


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
        model.add_output('choice', tf.squeeze(tf.multinomial(policy, 1), [1]))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        advantage = tf.stop_gradient(return_ - value)
        logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
        actor = config.actor_weight * advantage * logprob
        critic = config.critic_weight * (return_ - value) ** 2 / 2
        entropy = config.entropy_weight * entropy
        model.add_cost('cost', critic - actor - entropy)


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
        qvalues = model.add_output(
            'qvalues', tf.squeeze(dense(hidden, actions, tf.identity), [1]))
        policy = dense(value, actions, tf.nn.softmax)
        model.add_output('choice', tf.squeeze(tf.multinomial(policy, 1), [1]))
    with tf.variable_scope('learning'):
        action = model.add_input('action', type_=tf.int32)
        action = tf.one_hot(action, actions)
        return_ = model.add_input('return_')
        qvalue = qvalues * action
        advantage = tf.stop_gradient(qvalue - value)
        logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
        entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
        actor = config.actor_weight * advantage * logprob
        critic = config.critic_weight * (return_ - value) ** 2 / 2
        qcritic = config.critic_weight * (return_ - qvalue) ** 2 / 2
        entropy = config.entropy_weight * entropy
        model.add_cost('cost', critic - actor - entropy)
