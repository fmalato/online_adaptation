import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo import PPO
from imitation.policies.serialize import load_policy

from wrappers import TorchObsGymWrapper


class PreTrainedIDMEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, policy_type_gail, venv, model_path, feats_dim):
        super().__init__(
            observation_space=observation_space,
            features_dim=feats_dim
        )
        self.encoder = load_policy(
            policy_type=policy_type_gail,
            venv=venv,
            path=model_path
        )
        # Disable training
        self.encoder.eval()

    def forward(self, x):
        # Disable gradients update
        with torch.no_grad():
            return self.encoder.extract_features(x)


def main(args):
    os.makedirs(f"models/ppo/{args.env_name}", exist_ok=True)
    env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    env = TorchObsGymWrapper(env, history_length=args.history_length)

    net_arch = dict(pi=[256, 256], vf=[256, 256])
    feats_dim = 1024
    encoder_kwargs = dict(
        policy_type_gail="ppo",
        venv=env,
        model_path=args.encoder_path,
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
        learning_rate=1e-4,
        gamma=0.99,
        device="cuda:0",
        verbose=2,
        ent_coef=0.02,
        target_kl=0.03,
        gae_lambda=0.99
    )
    agent.set_random_seed(seed=np.random.randint(1, 100000))

    agent.learn(total_timesteps=args.timesteps)
    torch.save(agent.policy.state_dict(), f"models/ppo/{args.env_name}/{args.timesteps}_ts_{args.history_length}_hist_len.pth")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-env', '--env-name', required=True, type=str)
    arg_parser.add_argument('-d', '--encoder-path', required=True, type=str)
    arg_parser.add_argument('-t', '--timesteps', required=True, type=int)
    arg_parser.add_argument('-s', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-e', '--epochs', default=10, type=int)
    arg_parser.add_argument('-b', '--batch-size', default=64, type=int)
    arg_parser.add_argument('-hist', '--history-length', default=16, type=int)
    arg_parser.add_argument('-head', '--headless', action='store_true')

    args = arg_parser.parse_args()

    main(args)
