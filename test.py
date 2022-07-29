import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from envs import create_env, _process_frame, ExplorationGame
from model import ActorCritic
from utils import update_count_matrix, calculate_entropy, reset_intrinsic_param
import logging

def test(rank, args, shared_model):
    log_file_name = f'{args.env_name.split("/")[-1]}-{args.intrinsic_coef}.log'
    open(f'logs/{log_file_name}', 'w').close()
    logging.basicConfig(filename=f'logs/{log_file_name}', level=logging.DEBUG)

    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name)
    if 'gridworld' not in args.env_name:
        env.seed(args.seed + rank)

    model = ActorCritic(env.action_space)

    model.eval()

    total_count = 0
    last_entropy = 0
    count_matrix = np.zeros((40,40,1))
    search_idx = {0:0}

    state = env.reset()
    state = _process_frame(state, args.env_name)
    state = torch.from_numpy(state)
    game_lens = []
    reward_sum = 0
    reward_sums = []
    rewards_intrinsic = []
    done = True

    start_time = time.time()

    episode_length = 0
    paths = []

    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0).unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        if 'gridworld' in args.env_name or 'Montezuma' in args.env_name:
            action = prob.multinomial(num_samples=1).detach()
        else:
            action = prob.max(1, keepdim=True)[1].numpy()
        state, reward, done, info = env.step(action.item())
        if 'Montezuma' in args.env_name:
            done = done or episode_length >= args.max_episode_length or reward != 0
        else:
            done = done or episode_length >= args.max_episode_length

        if 'gridworld' not in args.env_name:
            reward_sum += reward
        else:
            reward_sum += info
        
        if done:
            if args.env_name == 'gridworld':
                paths.append(env.path)

            logging.info(f'{time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}, {reward_sum}, {current_entropy}, {episode_length}')
            reward_sums.append(reward_sum)
            game_lens.append(episode_length)
            rewards_intrinsic.append(current_entropy)
            if len(reward_sums) == 20:
                if args.env_name == 'gridworld':
                    total_paths = np.concatenate(np.asarray(paths))
                    np.save('total_paths', total_paths)
                    print('Paths saved', total_paths.shape)

                print(f'Time Trained = {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}')
                print(f'Mean Reward = {np.mean(reward_sums)}, Max Reward = {np.max(reward_sums)}, Game Length = {np.mean(game_lens)}')
                print(f'Final Entropy = {np.mean(rewards_intrinsic)}')
                torch.save(model.state_dict(), 'model.pth')
                #if np.mean(reward_sums) > best_reward:
                #    torch.save(model.state_dict(), 'model.pth')
                #    best_reward = np.mean(reward_sums)
                game_lens = []
                reward_sums = []
                rewards_intrinsic = []
            reward_sum = 0
            episode_length = 0
            total_count = 0
            last_entropy = 0
            count_matrix = np.zeros((40,40,1))
            search_idx = {0:0}
            state = env.reset()

        state = _process_frame(state, args.env_name)
        
        count_matrix, total_count, search_idx = update_count_matrix(count_matrix, state, total_count, search_idx)
        current_entropy = calculate_entropy(count_matrix, total_count)
        reward_intrinsic = (current_entropy-last_entropy)*args.intrinsic_coef
        last_entropy = current_entropy

        state = torch.from_numpy(state)
