# Deep-Q-Learning-Treasure-Hunt

Deep-Q-Learning-Treasure-Hunt is an enhanced version of the [Q-Learning-Treasure-Hunt](https://github.com/Ameya-Punekar/Q-Learning-Treasure-Hunt.git) project, incorporating Deep Q-Learning (DQN) to address the limitations of traditional Q-Learning methods such as scalability, memory consumption, and slow convergence. By leveraging neural networks, this project aims to improve the performance of an agent in a treasure hunt environment.

## File Structure

### `main.py`
The main script for training, testing, and optionally rendering the DQN agent.

**Key Responsibilities:**
- **Training**: Initializes the environment and DQN agent, trains the agent over specified episodes, and saves the trained model.
- **Testing**: Loads and evaluates the trained model's performance.
- **Rendering**: Optionally visualizes the agent's actions in the environment.

**Key Sections:**
- **Imports**: Essential libraries and custom modules.
- **Configuration**: Environment settings and hyperparameters.
- **Training Loop**: Manages action selection, experience replay, and model updates.
- **Testing**: Evaluates the trained agent’s performance.
- **Rendering Option**: Visualizes the environment during training/testing.

### `utils.py`
Utility functions for experience replay and training.

**Key Classes and Functions:**
- **ReplayBuffer**: Manages experience storage and sampling.
  - `__init__(self, buffer_limit)`: Initializes the buffer.
  - `put(self, transition)`: Adds a transition to the buffer.
  - `sample(self, n)`: Samples a mini-batch from the buffer.
  - `size(self)`: Returns the buffer size.
- **train(q_net, q_target, memory, optimizer, batch_size, gamma)**: Executes a training step using a mini-batch from the buffer.

### `DQN_model.py`
Defines the architecture of the Q-network for approximating the Q-function.

**Key Classes:**
- **Qnet**: Neural network model for the Q-function.
  - `__init__(self, no_actions, no_states)`: Initializes the network with hidden layers and an output layer.
  - `forward(self, x)`: Defines the forward pass with ReLU activation.
  - `sample_action(self, observation, epsilon)`: Selects an action using the epsilon-greedy policy.
