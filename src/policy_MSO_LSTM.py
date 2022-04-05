'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim'] + policy_params['latent_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)
        self.latent_dim = policy_params['latent_dim']
        self.policy_type = policy_params['type']
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LSTMPolicy(Policy):
  """LSTM policy."""

  def __init__(self, policy_params):
    """Initializes the lstm policy. See the base class for more details."""

    # Assume the first hidden layer is recurrent, the rest is feed forward
    # Policy.__init__(self, policy_params, update_filter=update_filter)
    Policy.__init__(self, policy_params)
    self._hidden_layer_sizes = policy_params["policy_network_size"]
    self._activation = policy_params.get("activation", "tanh")
    if self._activation == "tanh":
        self._activation = np.tanh
    elif self._activation == "clip":
        self._activation = lambda x: np.clip(x, -1.0, 1.0)

    self._lstm_weight_size = [self.ob_dim+self._hidden_layer_sizes[0], self._hidden_layer_sizes[0]]
    self._lstm_weight_start_idx = []
    self._lstm_weight_end_idx = []

    self._layer_sizes = []
    self._layer_sizes.extend(self._hidden_layer_sizes)
    self._layer_sizes.append(self.ac_dim)
    self._layer_weight_start_idx = []
    self._layer_weight_end_idx = []
    num_weights = 0

    # W_f, W_i, W_c, W_o
    for i in range(4):
        self._lstm_weight_start_idx.append(num_weights)
        num_weights += (
            self._lstm_weight_size[0] * self._lstm_weight_size[1])
        self._lstm_weight_end_idx.append(num_weights)

    num_layers = len(self._layer_sizes)
    for ith_layer in range(num_layers - 1):
        self._layer_weight_start_idx.append(num_weights)
        num_weights += (
          self._layer_sizes[ith_layer] * self._layer_sizes[ith_layer + 1])
        self._layer_weight_end_idx.append(num_weights)
    self.weights = np.zeros(num_weights, dtype=np.float64)

    self.hidden_c = np.zeros(self._lstm_weight_size[1])
    self.hidden_h = np.zeros(self._lstm_weight_size[1])

    if "weights" in policy_params:
      self.weights = np.zeros(num_weights, dtype = np.float64)
      weights_src = policy_params["weights"].copy()
      self.weights[:] = weights_src[:]
      # self.weights = policy_params["weights"]
    else:
#      self.weights = np.zeros(num_weights)
       self.weights = 0.01 * (np.random.rand(num_weights) - 0.5)

  def _sigmoid(self, x):
    return np.exp(-np.logaddexp(0, -x))

  def act(self, ob):
    """Maps the observation to action.
    Args:
      ob: The observations in reinforcement learning.
    Returns:
      actions: The actions in reinforcement learning.
    """
    # Compute the LSTM part
    W_f = np.reshape(self.weights[self._lstm_weight_start_idx[0]:self._lstm_weight_end_idx[0]],(self._lstm_weight_size[0], self._lstm_weight_size[1]))
    W_i = np.reshape(self.weights[self._lstm_weight_start_idx[1]:self._lstm_weight_end_idx[1]],(self._lstm_weight_size[0], self._lstm_weight_size[1]))
    W_c = np.reshape(self.weights[self._lstm_weight_start_idx[2]:self._lstm_weight_end_idx[2]],(self._lstm_weight_size[0], self._lstm_weight_size[1]))
    W_o = np.reshape(self.weights[self._lstm_weight_start_idx[3]:self._lstm_weight_end_idx[3]],(self._lstm_weight_size[0], self._lstm_weight_size[1]))
    ft = self._sigmoid(np.dot(np.concatenate([self.hidden_h, ob]), W_f))
    it = self._sigmoid(np.dot(np.concatenate([self.hidden_h, ob]), W_i))
    ct = np.tanh(np.dot(np.concatenate([self.hidden_h, ob]), W_c))
    self.hidden_c = ft * self.hidden_c + it * ct
    ot = self._sigmoid(np.dot(np.concatenate([self.hidden_h, ob]), W_o))
    self.hidden_h = ot * np.tanh(self.hidden_c)
    ith_layer_result = self.hidden_h

    # Compute the feed-forward part
    num_layers = len(self._layer_sizes)
    for ith_layer in range(num_layers - 1):
      mat_weight = np.reshape(
          self.weights[self._layer_weight_start_idx[ith_layer]:
                       self._layer_weight_end_idx[ith_layer]],
          (self._layer_sizes[ith_layer + 1], self._layer_sizes[ith_layer]))
      ith_layer_result = np.dot(mat_weight, ith_layer_result)
      ith_layer_result = self._activation(ith_layer_result)

    normalized_actions = ith_layer_result
    # actions = (
        # normalized_actions * (self.action_high - self.action_low) / 2.0 +
        # (self.action_low + self.action_high) / 2.0)
    return normalized_actions # actions

  def reset(self):
    self.hidden_c = np.zeros(self._lstm_weight_size[1])
    self.hidden_h = np.zeros(self._lstm_weight_size[1])

        