experiment: heads
timesteps: 3.2e7
epoch_length: 2e5
repeats: 2
envs:
  - SimplePong-v0
  - SimpleBreakout-v0
  - SimpleDeathmatch-v0
agents:
  -
    name: A3C (16)
    type: A3C
    heads: 16
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: network_a3c_lstm
  -
    name: A3C (32)
    type: A3C
    heads: 32
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: network_a3c_lstm
  -
    name: A3C (64)
    type: A3C
    heads: 64
    epsilon_from: 1.0
    epsilon_tos: [0.1, 0.01, 0.5]
    epsilon_duration: 6e5
    initial_learning_rate: 1e-4
    network: network_a3c_lstm
