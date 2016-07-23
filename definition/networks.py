experiment: networks
timesteps: 3.2e7
epoch_length: 2e5
repeats: 2
envs:
  - SimplePong-v0
  - SimpleDeathmatch-v0
agents:
  -
    name: A3C (DRQN)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_drqn'
  -
    name: A3C (A3C-LSTM)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_a3c_lstm'
  -
    name: A3C (Doom L)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_doom_large'
  -
    name: A3C (Minecraft S)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_minecraft_small'
  -
    name: A3C (My FF 1)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_1'
  -
    name: A3C (My RNN 2)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_2'
  -
    name: A3C (My RNN 3)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: 'network_3'
