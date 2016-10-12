import tensorflow as tf
from mindpark.model.layer import dense, conv2d, rnn


def q_learning(model, network, observs, actions, config=None):
    """
    Bootstrapped action value approximation.
    """
    # Percetion.
    state = model.add_input('state', observs)
    hidden = network(model, state)
    values = dense(hidden, actions, tf.identity)
    values = model.add_output('values', values)
    # Training.
    action = model.add_input('action', type_=tf.int32)
    action = tf.one_hot(action, actions)
    target = model.add_input('target')
    model.add_output('value', tf.reduce_max(values, 1))
    model.add_cost(
        'cost', (tf.reduce_sum(action * values, 1) - target) ** 2)


def actor_critic(model, network, observs, actions, config):
    """
    Bootstrapped policy gradient with value function baseline.
    """
    # Perception.
    state = model.add_input('state', observs)
    hidden = network(model, state)
    value = model.add_output(
        'value', tf.squeeze(dense(hidden, 1, tf.identity), [1]))
    policy = dense(value, actions, tf.nn.softmax)
    model.add_output(
        'choice', tf.squeeze(tf.multinomial(tf.log(policy), 1), [1]))
    # Training.
    action = model.add_input('action', type_=tf.int32)
    action = tf.one_hot(action, actions)
    return_ = model.add_input('return_')
    logprob = tf.log(tf.reduce_sum(policy * action, 1) + 1e-13)
    entropy = -tf.reduce_sum(tf.log(policy + 1e-13) * policy)
    advantage = tf.stop_gradient(return_ - value)
    actor = advantage * logprob + config.regularize * entropy
    critic = config.scale_critic_loss * (return_ - value) ** 2 / 2
    model.add_cost('cost', critic - actor)
