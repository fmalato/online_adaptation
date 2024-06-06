import argparse
import os

import numpy as np
import gymnasium as gym
import miniworld
import torch
import faiss

from tqdm import tqdm
from imitation.policies.serialize import load_policy

from test_bc import ActorCriticPolicy
from resnet_encoder import CausalIDMEncoder
from utils import stack_obs_frames, constant_lr
from wrappers import TorchObsGymWrapper


class ZIPAgent:
    def __init__(self, embeddings_path, max_followed=16, divergence_scaling_factor=2.0, history_length=16, debug=False,
                 grayscale=True, encoder=None, normalize_input=True, use_faiss=True, index_dim=1024):
        self.grayscale = grayscale
        self.encoder = encoder
        self.normalize_input = normalize_input
        self.embeddings = None
        self.first = True
        self.current_idx = 0
        self.actions = None
        self.reference_embed = None
        self.reference_distance = 0.0
        self.history_length = history_length
        self.max_followed = max_followed
        self.followed_count = 0
        self.divergence_scaling_factor = divergence_scaling_factor
        self.debug = debug
        self.use_faiss = use_faiss
        self.index_dim = index_dim
        if self.use_faiss:
            self.embed_index = faiss.IndexFlatL2(self.index_dim)
        else:
            self.embed_index = None
        self.load_embeddings(embeddings_path)

    def load_embeddings(self, embeddings_path):
        trajectories = [x for x in os.listdir(embeddings_path) if ".npz" in x]
        assert len(trajectories) > 0, "No trajectories to analyze!"

        observations = []
        actions = []
        for t in trajectories:
            trajectory_path = os.path.join(embeddings_path, t)
            data = np.load(trajectory_path, allow_pickle=True)
            if self.normalize_input:
                o = data["observations"].astype(np.float32) / 255
            else:
                o = data["observations"].astype(np.float32)
            if self.grayscale and self.encoder is None:
                o = 0.2989 * o[:, :, :, 0] + 0.5870 * o[:, :, :, 1] + 0.1140 * o[:, :, :, 2]
                observations.append(np.reshape(o, newshape=(o.shape[0], -1)))
            elif self.encoder is not None:
                if self.history_length > 1:
                    o = stack_obs_frames(o, history_length=self.history_length)
                with torch.no_grad():
                    o = self.encoder.extract_features(torch.FloatTensor(np.transpose(o, (0, 1, 4, 2, 3))).cuda())
                observations.append(o.cpu().numpy())
            else:
                observations.append(np.reshape(o, newshape=(o.shape[0], -1)))
            actions.append(data["actions"])

        self.embeddings = np.concatenate(observations, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        if self.use_faiss:
            self.embed_index.add(self.embeddings)

    def predict(self, obs, deterministic):
        return self.get_action(obs), {}

    def get_action(self, obs):
        if self.grayscale and self.encoder is None:
            if obs.shape == (3, 60, 80):
                obs = np.transpose(obs, (2, 0, 1))
            obs = 0.2989 * obs[:, :, 0] + 0.5870 * obs[:, :, 1] + 0.1140 * obs[:, :, 2]
        elif self.encoder is not None:
            with torch.no_grad():
                obs = self.encoder.extract_features(torch.FloatTensor(np.array(obs)).unsqueeze(0).cuda())
                obs = obs.cpu().numpy()
        if self.use_faiss:
            distance, idx = self.embed_index.search(obs, k=1)
            if len(idx[0]) == 1:
                current_distance = float(distance)
            else:
                probs = (1 / distance[0]) / np.sum(1 / distance[0])
                curr = np.argwhere(idx[0] == np.random.choice(idx[0], p=probs))
                current_distance = float(distance[0][curr])
        else:
            difference_matrix = np.sum(np.abs(self.embeddings - obs.flatten()), axis=1)
            current_distance = np.min(difference_matrix, axis=0)

        if self.debug:
            if self.first:
                print("First-step search.")
            elif self.followed_count >= self.max_followed - 1:
                print("Time-based search.")
            elif current_distance > self.divergence_scaling_factor * self.reference_distance:
                print("Divergence-based search.")
            elif self.current_idx + self.followed_count >= self.actions.shape[0] - 1:
                print("Action-overflow search.")
            print(f"Current index: {self.current_idx} - Current distance: {current_distance}")

        if self.first or self.followed_count >= self.max_followed - 1 or \
            current_distance > self.divergence_scaling_factor * self.reference_distance or \
                self.current_idx + self.followed_count >= self.actions.shape[0] - 1:
            self.followed_count = 0
            if self.use_faiss:
                self.current_idx = int(idx) if len(idx[0]) == 1 else int(idx[0][curr])
            else:
                self.current_idx = np.argmin(difference_matrix, axis=0)
            self.reference_distance = current_distance
            self.reference_embed = obs.flatten()
            if self.first:
                self.first = False
        else:
            self.followed_count += 1

        return self.actions[self.current_idx + self.followed_count]

    def search_index(self, o, k=10):
        distances, indices = self.embed_index.search(o, k=k)
        return distances, indices

    def reset(self):
        self.followed_count = 0
        self.reference_distance = 0.0
        self.reference_embed = None
