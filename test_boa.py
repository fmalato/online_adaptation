import argparse
import os

import numpy as np
import torch
import gymnasium as gym
import miniworld

from imitation.policies.serialize import load_policy
from tqdm import tqdm
from stable_baselines3.common.policies import ActorCriticPolicy

from boa_agent import HybridAdaptedAgent
from utils import constant_lr
from resnet_encoder import CausalIDMEncoder
from wrappers import TorchObsGymWrapper


def test(args):
    env = gym.make(args.env_name, render_mode="human", max_episode_steps=args.max_steps)
    env = TorchObsGymWrapper(env, history_length=args.history_length)

    net_arch = dict(pi=[256, 256], vf=[256, 256])
    feats_dim = 1024
    lr_schedule = "constant"

    if args.il_agent_type == "gail":
        il_agent = load_policy("ppo", env, path=f"models/{args.il_agent_type}/{args.env_name}/{args.il_agent_name}")
    else:
        idm_encoder_kwargs = dict(
            feats_dim=feats_dim,
            conv3d_in_channels=args.history_length,
            conv3d_out_channels=128,
            resnet_in_channels=[128, 64, 128],
            resnet_out_channels=[64, 128, 128],
            input_size=(1, args.history_length, 3, 60, 80),
            use_conv3d=True,
            device="cuda"
        )
        il_agent = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            features_extractor_class=CausalIDMEncoder,
            features_extractor_kwargs=idm_encoder_kwargs,
            activation_fn=torch.nn.ReLU,
            net_arch=net_arch,
            lr_schedule=constant_lr if lr_schedule == "constant" else None
        )
        il_agent.load_state_dict(torch.load(f"models/{args.il_agent_type}/{args.env_name}/{args.il_agent_name}"))

    agent = HybridAdaptedAgent(
        embeddings_path=f"trajectories/{args.env_name}",
        il_agent=il_agent,
        num_actions=env.action_space.n,
        log=args.log,
        il_agent_type=args.il_agent_type,
        k=args.num_queries,
        index_dim=feats_dim,
        history_length=args.history_length,
        deterministic=args.deterministic,
        env_name=args.env_name
    )

    os.makedirs(f"results/{args.env_name}", exist_ok=True)
    with open(f"results/{args.env_name}/hybrid_agent_test_results.txt", "a+") as f:
        if args.il_agent_type == "bc":
            agent_type = "BC"
        elif args.il_agent_type == "gail":
            agent_type = "GAIL"
        else:
            agent_type = "IL"

        if args.deterministic:
            det = "DETERMINISTIC"
        else:
            det = "STOCHASTIC"
        f.write(f"NEW RUN - {det} - [RESNET ENCODER ({agent_type})] - model: {args.il_agent_name} - k={args.num_queries}\n")

    for r in range(args.num_test_runs):
        num_timesteps = []
        success = []
        ep_rewards = []
        progress_bar = tqdm(range(args.num_games))
        for i in progress_bar:
            observation, info = env.reset()
            terminated = False
            truncated = False
            timestep = 0
            obs = []
            acts = []
            ep_reward = 0.0
            while not (terminated or truncated):
                action = agent.get_action(observation)
                if args.record:
                    obs.append(observation[args.history_length - 1])
                    acts.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if args.show:
                    env.render()
                timestep += 1

                if terminated or truncated:
                    num_timesteps.append(timestep)
                    success.append(1 if (terminated and ep_reward > 0.0) else 0)

            agent.set_episode_number(n=i + 1)
            ep_rewards.append(ep_reward)

            progress_bar.set_description(f"Avg. number of steps: {np.mean(num_timesteps):.3f} +/- {np.std(num_timesteps):.3f} | Success percentage: {np.mean(success) * 100:.3f}% | Avg. Reward: {np.mean(ep_rewards):.3f} +/- {np.std(ep_rewards):.3f}")

            if args.record:
                os.makedirs(f"recorded/{args.env_name}/BOA_{agent_type}/", exist_ok=True)
                np.savez_compressed(f"recorded/{args.env_name}/BOA_{agent_type}/game_{i}.npz", observations=np.array(obs), actions=np.array(acts))

        env.close()
        avg_steps = np.mean(num_timesteps)
        std_steps = np.std(num_timesteps)
        success_percentage = np.mean(success) * 100
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        with open(f"results/{args.env_name}/hybrid_agent_test_results.txt", "a+") as f:
            f.write(f"Avg. number of steps: {avg_steps:.3f} +/- {std_steps:.3f} | Success percentage: {success_percentage:.3f}% | Avg. Reward: {avg_reward:.3f} +/- {std_reward:.3f}\n")
        f.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-g', '--env-name', required=True, type=str)
    arg_parser.add_argument('-n', '--num-games', required=True, type=int)
    arg_parser.add_argument('-p', '--il-agent-name', required=True, type=str)
    arg_parser.add_argument('-bcm', '--il-agent-type', required=True, type=str)
    arg_parser.add_argument('-f', '--history-length', default=16, type=int)
    arg_parser.add_argument('-k', '--num-queries', default=10, type=int)
    arg_parser.add_argument('-s', '--max-steps', default=1000, type=int)
    arg_parser.add_argument('-det', '--deterministic', action="store_true")
    arg_parser.add_argument('-b', '--debug', action="store_true")
    arg_parser.add_argument('-l', '--log', action="store_true")
    arg_parser.add_argument('-r', '--record', action="store_true")
    arg_parser.add_argument('-sh', '--show', action="store_true")
    arg_parser.add_argument('-tr', '--num-test-runs', default=1, type=int)

    args = arg_parser.parse_args()

    test(args)
