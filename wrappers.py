from typing import SupportsFloat, Any

import numpy as np
import gymnasium as gym
import gymnasium.spaces.box as box

from gymnasium.core import WrapperActType, WrapperObsType


class TorchObsGymWrapper(gym.Wrapper):
    def __init__(self, env, history_length=1):
        super().__init__(env=env)
        obs_shape = env.observation_space.shape
        self.observation_space = box.Box(low=0, high=1, shape=(history_length, obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32)
        self.history_length = history_length
        self.obs_buffer = [np.zeros(shape=self.observation_space.shape[1:]) for i in range(history_length)]

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs.transpose((2, 0, 1)).astype(np.float32) / 255)

        return self.obs_buffer, reward, terminated, truncated, info

    def reset(
        self, *, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset()
        self.obs_buffer = [np.zeros(shape=self.observation_space.shape[1:]) for i in range(self.history_length)]
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs.transpose((2, 0, 1)).astype(np.float32) / 255)

        return self.obs_buffer, info

    def seed(self, s):
        super().seed(s)
