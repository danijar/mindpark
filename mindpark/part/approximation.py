import tensorflow as tf
from mindpark.model.layer import dense, conv2d, rnn


def q_function(model, network, observs, actions, config=None):
    """
    Action value approximation.
    """
    # Percetion.
    state = model.add_input('state', observs)
    hidden = network(model, state)
    values = dense(hidden, actions, tf.identity)
    values = model.add_output('values', values)
    # Training.
    action = model.add_input('action', type_=tf.int32)
    action = tf.one_hot(action, actions)
    return_ = model.add_input('return_')
    model.add_output('value', tf.reduce_max(values, 1))
    model.add_cost(
        'cost', (tf.reduce_sum(action * values, 1) - return_) ** 2)


def value_policy_gradient(model, network, observs, actions, config):
    """
    Policy gradient with value function baseline.
    """
    # perception.
    state = model.add_input('state', observs)
    hidden = network(model, state)
    value = model.add_output(
        'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
    policy = dense(value, actions, tf.nn.softmax)
    model.add_output(
        'choice', tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
    # training.
    action = model.add_input('action', type_=tf.int32)
    action = tf.one_hot(action, actions)
    return_ = model.add_input('return_')
    logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
    entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
    advantage = tf.stop_gradient(return_ - value)
    actor = advantage * logprob + config.regularize * entropy
    critic = config.scale_critic_loss * (return_ - value) ** 2 / 2
    model.add_cost('cost', critic - actor)
