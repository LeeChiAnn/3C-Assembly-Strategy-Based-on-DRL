import argparse
import numpy as np
import torch
from sac import SAC
from replay_memory import ReplayMemory
import envs.sucker as sucker
from gym.spaces import Box

# 构建动作空间
num_states = len(sucker.get_obs())
print("Size of State Space ->  {}".format(num_states))
action_space = Box(low=-0.6, high=0.6, shape=(7,), dtype=np.float64)
action_space.high = np.array([0.6, 0.15, 0.3, 0.3, 0.3, 3.14, 1])
action_space.low = np.array([-0.6, -0.15, -0.3, -0.3, -0.3, -3.14, 0])
num_actions = action_space.shape
print("Size of Action Space ->  {}".format(num_actions))
upper_bound = action_space.high
lower_bound = action_space.low
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)    # the st seed, the same number of seed, the random is same

# Agent
agent = SAC(num_states, action_space, args)
agent.load_model('models/sac_actor_sucker_', 'models/sac_critic_sucker_')

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0

for i_episode in range(1000):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = sucker.reset()

    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = sucker.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        state = next_state

    if total_numsteps > args.num_steps:
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                      episode_steps,
                                                                                      round(episode_reward, 2)))
        break
sucker.v.close()

