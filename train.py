import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from envs import create_env, _process_frame, ExplorationGame
from model import ActorCritic
from utils import ensure_shared_grads, update_count_matrix, calculate_entropy, reset_intrinsic_param


def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name)
    if 'gridworld' not in args.env_name:
        env.seed(args.seed + rank)

    model = ActorCritic(env.action_space)

    model.train()

    total_count, last_entropy, count_matrix, search_idx = reset_intrinsic_param()
    reward_extrinsic = 0
    
    state = env.reset()
    state = _process_frame(state, args.env_name)
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0).unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            
            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)
            state, game_reward, done, _ = env.step(action.item())
            if 'Montezuma' in args.env_name:
                done = done or episode_length >= args.max_episode_length or game_reward != 0
            else:
                done = done or episode_length >= args.max_episode_length
            if 'Mario' in args.env_name:
                game_reward = 0 # game_reward/2
            reward_extrinsic += game_reward

            if done:
                episode_length = 0
                state = env.reset()
            
            state = _process_frame(state, args.env_name)

            count_matrix, total_count, search_idx = update_count_matrix(count_matrix, state, total_count, search_idx)
            current_entropy = calculate_entropy(count_matrix, total_count)
            reward_intrinsic = (current_entropy-last_entropy)*args.intrinsic_coef
            last_entropy = current_entropy

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            if done:
                total_count, last_entropy, count_matrix, search_idx = reset_intrinsic_param()
                # rewards.append(reward_extrinsic + reward_intrinsic)             # uncomment this for Pong Sparse Reward
                reward_extrinsic = 0
            else:
                # rewards.append(0 + reward_intrinsic)                            # uncomment this for Pong Sparse Reward
                pass                                                              # comment this for Pong Sparse Reward
            rewards.append(game_reward + reward_intrinsic)                        # comment this for Pong Sparse Reward

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0).unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
