# Imports:
# --------
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from padm_env import create_env

# User definitions:
# -----------------
train_dqn = False
test_dqn = True
render = True

# Define environment attributes (environment specific):
goal_coordinates = (6, 6)
obstacle_coordinates = [(0, 3), (1, 2), (2, 3), (3, 1), (4, 4), (6, 2)]
friendly_state_coordinates = [(1, 0), (3, 4), (4,0), (5, 1), (5,3), (5,6), (6,5)]
non_friendly_state_coordinates = [(0, 1), (2,1),(4,2),(6, 0)]

# Hyperparameters:
# ----------------
learning_rate = 0.005
gamma = 0.998
buffer_limit = 100_000
batch_size = 64
num_episodes = 15000  
max_steps = 5000  

# Custom epsilon decay strategy:
def custom_epsilon_decay(n_epi, total_episodes, min_epsilon=0.1):
    epsilon_start = 1.0  # Start with full exploration
    epsilon_end = min_epsilon  # Minimum value of epsilon
    epsilon_decay = (epsilon_start - epsilon_end) / total_episodes
    epsilon = epsilon_start - epsilon_decay * n_epi
    return max(epsilon, epsilon_end)

# Main:
# -----
if train_dqn:
    env = create_env(goal_coordinates, obstacle_coordinates, friendly_state_coordinates, non_friendly_state_coordinates)

    q_net = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    q_target = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    q_target.load_state_dict(q_net.state_dict())

    memory = ReplayBuffer(buffer_limit=buffer_limit)
    print_interval = 100
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    rewards = []

    for n_epi in range(num_episodes):
        epsilon = custom_epsilon_decay(n_epi, num_episodes)
        s, _ = env.reset()
        done = False

        for _ in range(max_steps):
            if render:
                env.render()
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            episode_reward += r

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0

    env.close()
    torch.save(q_net.state_dict(), "dqn.pth")
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env(goal_coordinates, obstacle_coordinates, friendly_state_coordinates, non_friendly_state_coordinates)

    dqn = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            if render:
                env.render()
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _ = env.step(action.argmax().item())
            s = s_prime
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
