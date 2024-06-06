import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time

import gymnasium as gym
import keyboard
import numpy as np
import miniworld

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


def on_press(key):
    pass

def on_release(key):
    return False, key


def map_action(key):
    action = 6
    if key == "a":
        action = 0
    elif key == "d":
        action = 1
    elif key == "w":
        action = 2
    elif key == "s":
        action = 3
    elif key == "e":
        action = 4
    elif key == "q":
        action = 5

    return action


def record(args):
    os.makedirs(args.save_path, exist_ok=True)
    # If previous games have been saved, avoid overlapping names for trajectories
    try:
        num_previous_games = len(os.listdir(args.save_path))
    except Exception:
        num_previous_games = 0

    env = gym.make(args.env_name, render_mode="human")
    if "MiniGrid" in args.env_name:
        env = ImgObsWrapper(RGBImgPartialObsWrapper(env))

    for i in range(args.num_games):
        observation, info = env.reset()
        terminated = False
        observations = []
        actions = []
        while not terminated:
            key = keyboard.read_key(suppress=True)
            if key == "p":
                from PIL import ImageGrab
                im = ImageGrab.grab()
                t = time.time_ns()
                im.save(f"{args.env_name}_{t}.png")
            action = map_action(key)
            observations.append(observation)
            actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
        np.savez_compressed(f"{args.save_path}/{i + num_previous_games}.npz", observations=np.array(observations), actions=np.array(actions))

    env.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-n', '--num-games', required=True, type=int)
    arg_parser.add_argument('-g', '--env-name', required=True, type=str)
    arg_parser.add_argument('-s', '--save-path', required=True, type=str)

    args = arg_parser.parse_args()

    record(args)
