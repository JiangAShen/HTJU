"""PPO Training with HTJU / Adam / SGD optimizers.

Environments: Pendulum-v1, Ant-v5, HalfCheetah-v5, Humanoid-v5

Usage:
    python main.py --env Pendulum-v1 --optimizer htju --epochs 500
    python main.py --env HalfCheetah-v5 --optimizer adam --epochs 1000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import gymnasium as gym
import torch
import numpy as np
import argparse
import time

from ppo import PPO


def get_args():
    p = argparse.ArgumentParser(description='PPO Training with HTJU Optimizers')
    p.add_argument('--env', type=str, default='Ant-v5',
                   choices=['Pendulum-v1', 'Ant-v5', 'HalfCheetah-v5', 'Walker2d-v5'])
    p.add_argument('--optimizer', type=str, default='htju',
                   choices=['htju', 'adam', 'sgd'])
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--max_step', type=int, default=200)
    p.add_argument('--batch_steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


# Per-environment defaults (HTJU paper Table D.2)
ENV_DEFAULTS = {
    'Pendulum-v1':         {'epochs': 500,  'max_step': 200, 'batch_steps': 2000},
    'Ant-v5':              {'epochs': 80000, 'max_step': 1000, 'batch_steps': 2000},
    'HalfCheetah-v5':      {'epochs': 75000, 'max_step': 1000, 'batch_steps': 2000},
    'Walker2d-v5':         {'epochs': 150000, 'max_step': 1000, 'batch_steps': 2000},
}


def main():
    args = get_args()
    cfg = ENV_DEFAULTS[args.env]

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim, opt_name=args.optimizer,
                total_epochs=cfg['epochs'])

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.env}_{args.optimizer}_seed{seed}_{timestamp}.txt"
    os.makedirs('txt_save', exist_ok=True)
    log_filepath = os.path.join('txt_save', log_filename)

    print(f"Training: {args.env} | {args.optimizer} | epochs={cfg['epochs']} | seed={seed}")
    print(f"Saving to: {log_filepath}")

    with open(log_filepath, "w") as f:
        for epo in range(cfg['epochs']):
            memory, scores, steps = [], [], 0

            while steps < cfg['batch_steps']:
                state, _ = env.reset(seed=seed + epo)
                score = 0
                for _ in range(cfg['max_step']):
                    steps += 1
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = agent.actor_net.choose_action(state_t)[0]
                    real_action = np.clip(action * 2.0, -2.0, 2.0)
                    next_state, reward, done, trunc, _ = env.step(real_action)

                    # Pendulum: shaped reward toward upright
                    if args.env == 'Pendulum-v1':
                        cos_theta = next_state[0]
                        reward = (cos_theta + 1.0) ** 2 - 0.1 * np.abs(next_state[2])

                    memory.append([state, action, reward, 0 if (done or trunc) else 1])
                    score += reward
                    state = next_state
                    if done or trunc:
                        break
                scores.append(score)

            # Advance HTJU epoch counter for s-weight schedule
            if hasattr(agent.actor_optim, 'epoch_now'):
                agent.actor_optim.epoch_now += 1
            if hasattr(agent.critic_optim, 'epoch_now'):
                agent.critic_optim.epoch_now += 1

            avg_score = np.mean(scores)
            f.write(f"{avg_score}\n")
            f.flush()

            print(f'Epoch {epo:3d}/{cfg["epochs"]} | Avg Score: {avg_score:.4f}')
            agent.train(memory, epo)

    print(f"Training complete. Log: {log_filename}")


if __name__ == '__main__':
    main()
