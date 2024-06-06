import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import torch
import numpy as np

from tqdm import tqdm
from stable_baselines3.ppo import PPO

from wrappers import TorchObsGymWrapper
from train_ppo import PreTrainedIDMEncoder


SEED = 42


def main(args):
    with open(f"results/{args.env_name}/ppo_test_results.txt", "a+") as f:
        if args.deterministic:
            f.write(f"NEW RUN - DETERMINISTIC - MODEL: {args.model_name}\n")
        else:
            f.write(f"NEW RUN - STOCHASTIC - MODEL: {args.model_name}\n")
    f.close()

    env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    env = TorchObsGymWrapper(env, history_length=args.history_length)

    net_arch = dict(pi=[256, 256], vf=[256, 256])
    feats_dim = 1024
    encoder_kwargs = dict(
        policy_type_gail="ppo",
        venv=env,
        model_path=f"models/gail/{args.env_name}/{args.encoder_name}",
        feats_dim=feats_dim
    )

    policy_kwargs = dict(
        features_extractor_class=PreTrainedIDMEncoder,
        features_extractor_kwargs=encoder_kwargs,
        net_arch=net_arch,
        activation_fn=torch.nn.ReLU
    )

    agent = PPO(
        env=env,
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=5e-5,
        gamma=0.99,
        device="cuda:0",
        verbose=2
    )

    agent.policy.load_state_dict(torch.load(f"models/ppo/{args.env_name}/{args.model_name}"))

    for n in range(args.num_test_runs):
        success = []
        num_timesteps = []
        ep_rewards = []
        progress_bar = tqdm(range(args.num_games))
        for i in progress_bar:
            obs, _ = env.reset(seed=SEED)
            ep_reward = 0.0
            timestep = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = agent.predict(obs, deterministic=args.deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                timestep += 1

                if terminated or truncated:
                    num_timesteps.append(timestep)
                    success.append(1 if (terminated and ep_reward > 0.0) else 0)

                if args.show:
                    env.render()

            ep_rewards.append(ep_reward)
            progress_bar.set_description(f"Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")

        avg_steps = np.mean(num_timesteps)
        std_steps = np.std(num_timesteps)
        success_percentage = np.mean(success) * 100
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)

        with open(f"results/{args.env_name}/ppo_test_results.txt", "a+") as f:
            f.write(
                f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
        f.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-env', '--env-name', required=True, type=str)
    arg_parser.add_argument('-d', '--encoder-name', required=True, type=str)
    arg_parser.add_argument('-m', '--model-name', required=True, type=str)
    arg_parser.add_argument('-n', '--num-games', required=True, type=int)
    arg_parser.add_argument('-ntr', '--num-test-runs', default=1, type=int)
    arg_parser.add_argument('-st', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-e', '--epochs', default=10, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=64, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-head', '--headless', action='store_true')
    arg_parser.add_argument('-s', '--show', action='store_true')
    arg_parser.add_argument('-det', '--deterministic', action='store_true')

    args = arg_parser.parse_args()

    main(args)
