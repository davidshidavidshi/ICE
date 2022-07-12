from __future__ import print_function

import argparse
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_env
from model import ActorCritic
from test import test
from train import train

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gae-lambda', type=float, default=1.00)
parser.add_argument('--entropy-coef', type=float, default=0.01)
parser.add_argument('--value-loss-coef', type=float, default=0.5)
parser.add_argument('--max-grad-norm', type=float, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num-processes', type=int, default=8)
parser.add_argument('--num-steps', type=int, default=20)
parser.add_argument('--max-episode-length', type=int, default=4000)
# ALE/MontezumaRevenge-v5, gridworld, PongDeterministic-v4, gridworldwall, BreakoutDeterministic-v4
parser.add_argument('--env-name', default='BreakoutDeterministic-v4')
parser.add_argument('--intrinsic-coef', type=float, default=1)
parser.add_argument('--load-model', default=False)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = create_env(args.env_name)
    shared_model = ActorCritic(env.action_space)
    if args.load_model:
        shared_model.load_state_dict(torch.load('model.pth'))
    shared_model.share_memory()

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()
    
    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)
    
    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
