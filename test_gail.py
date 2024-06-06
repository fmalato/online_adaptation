import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import torch

import gymnasium as gym
import miniworld
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy, CnnPolicy
from stable_baselines3.common.policies import NatureCNN
from tqdm import tqdm

from imitation.policies.serialize import load_policy

from wrappers import TorchObsGymWrapper


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    os.makedirs(f"results/{args.env_name}/", exist_ok=True)
    with open(f"results/{args.env_name}/gail_test_results.txt", "a+") as f:
        if args.deterministic:
            f.write(f"NEW RUN - DETERMINISTIC - MODEL: {args.model_name}\n")
        else:
            f.write(f"NEW RUN - STOCHASTIC - MODEL: {args.model_name}\n")
    f.close()

    env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    env = TorchObsGymWrapper(env, history_length=args.history_length)

    if args.model_name.endswith(".zip"):
        trained_agent = load_policy("ppo", env, path=f"models/gail/{args.env_name}/{args.model_name}")
    else:
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        feats_dim = 512
        policy_type = "CnnPolicy"
        batch_size = 64
        gamma = 0.99
        lr = 1e-4
        policy_kwargs = dict(
            features_extractor_class=NatureCNN,
            features_extractor_kwargs=dict(features_dim=feats_dim),
            net_arch=net_arch,
            activation_fn=torch.nn.ReLU
        )
        trained_agent = PPO(
            env=env,
            policy=CnnPolicy if policy_type == "CnnPolicy" else MlpPolicy,
            batch_size=batch_size,
            ent_coef=0.02,
            learning_rate=lr,
            gamma=gamma,
            n_epochs=10,
            seed=SEED,
            policy_kwargs=policy_kwargs
        )
        trained_agent.policy.load_state_dict(torch.load(f"models/gail/{args.env_name}/{args.model_name}").state_dict())

    for r in range(args.num_test_runs):
        num_timesteps = []
        success = []
        ep_durations = []
        ep_rewards = []
        progress_bar = tqdm(range(args.eval_episodes))
        progress_bar.set_description(f"[RUN {r + 1}/{args.num_test_runs}] Testing {args.model_name}")
        for i in progress_bar:
            observation, _ = env.reset(seed=SEED)
            terminated = False
            truncated = False
            timestep = 0
            ep_reward = 0.0
            if args.record:
                episode_obs = []
                episode_actions = []
                episode_obs.append(observation[args.history_length - 1])
            while not (terminated or truncated):
                action, _ = trained_agent.predict(observation, deterministic=args.deterministic)
                observation, reward, terminated, truncated, info = env.step(action)
                if args.record:
                    episode_actions.append(action)
                    episode_obs.append(observation[args.history_length - 1])
                ep_reward += reward
                if args.show:
                    env.render()
                timestep += 1

                if terminated or truncated:
                    num_timesteps.append(timestep)
                    success.append(1 if (terminated and ep_reward > 0.0) else 0)

            ep_durations.append(timestep)
            ep_rewards.append(ep_reward)
            progress_bar.set_description(f"Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")
            if args.record:
                os.makedirs(f"recorded/{args.env_name}/GAIL", exist_ok=True)
                np.savez_compressed(f"recorded/{args.env_name}/GAIL/game_{i}.npz", observations=np.array(episode_obs), actions=np.array(episode_actions))

        avg_steps = np.mean(num_timesteps)
        std_steps = np.std(num_timesteps)
        success_percentage = np.mean(success) * 100
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        with open(f"results/{args.env_name}/gail_test_results.txt", "a+") as f:
            f.write(
                f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
        f.close()
    env.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-d', '--model-name', required=True, type=str)
    arg_parser.add_argument('-e', '--env-name', required=True, type=str)
    arg_parser.add_argument('-t', '--eval-episodes', required=True, type=int)
    arg_parser.add_argument('-r', '--num-test-runs', default=3, type=int)
    arg_parser.add_argument('-st', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-s', '--show', action='store_true')
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-det', '--deterministic', action='store_true')
    arg_parser.add_argument('-rec', '--record', action='store_true')

    args = arg_parser.parse_args()

    main(args)
