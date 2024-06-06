import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import faiss

from datetime import datetime
from utils import stack_obs_frames


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HybridAdaptedAgent:
    def __init__(self, embeddings_path, il_agent, index_dim, num_actions, k=10, history_length=16, log=False,
                 il_agent_type="bc", device="cuda", deterministic=False, env_name="test", max_num_traj=-1):
        assert il_agent is not None, "HybridAdaptedAgent needs an IL agent!"
        self.device = device
        self.il_agent = il_agent
        self.il_agent.to(self.device)
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.first = True
        self.current_idx = 0
        self.actions = None
        self.reference_embed = None
        self.deterministic = deterministic
        self.reference_distance = 0.0
        self.history_length = history_length
        self.il_agent_type = il_agent_type
        self.index_dim = index_dim
        self.k = k
        self.max_num_traj = max_num_traj
        self.embed_index = faiss.IndexFlatL2(self.index_dim)
        self.load_embeddings(self.embeddings_path)
        self.num_actions = num_actions
        # Logging purposes
        self.log = log
        self.episode_number = 0
        self.env_name = env_name
        if self.log:
            self.current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            os.makedirs(f"logs/{self.env_name}/{self.current_time}", exist_ok=True)
            with open(f"logs/{self.env_name}/{self.current_time}/game_{self.episode_number}.log", "a+") as f:
                f.write("prior;zip_distrib;posterior;action;k\n")


    def load_embeddings(self, embeddings_path):
        if self.max_num_traj == -1:
            trajectories = [x for x in os.listdir(embeddings_path)[:self.max_num_traj] if ".npz" in x]
        else:
            trajectories = [x for x in os.listdir(embeddings_path) if ".npz" in x]
        assert len(trajectories) > 0, "No trajectories to analyze!"

        observations = []
        actions = []
        for t in trajectories:
            trajectory_path = os.path.join(embeddings_path, t)
            data = np.load(trajectory_path, allow_pickle=True)
            o = data["observations"].astype(np.float32) / 255
            o = stack_obs_frames(o, history_length=self.history_length)
            with torch.no_grad():
                o = self.il_agent.extract_features(torch.FloatTensor(np.transpose(o, (0, 1, 4, 2, 3))).to(self.device))
                observations.append(o.cpu().numpy())
            actions.append(data["actions"])

        self.embeddings = np.concatenate(observations, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.embed_index.add(self.embeddings)

    def predict(self, obs, deterministic=False):
        return self.get_action(obs), {}

    def get_action(self, obs):
        with torch.no_grad():
            o = self.il_agent.extract_features(torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device))
            o = o.cpu().numpy()
            distances, indices = self.embed_index.search(o, k=self.k)
            prior = self.il_agent.get_distribution(torch.FloatTensor(np.array(obs)).unsqueeze(0).to(self.device)).distribution.probs.cpu().numpy()[0]
            # Scaling factor on prior
            prior *= self.k
            zip_freq = self._get_frequencies(acts=self.actions[indices][0], normalize=False)
            alpha_new = torch.Tensor(prior + zip_freq) + 1e-7
            posterior = torch.distributions.Dirichlet(concentration=alpha_new)
            if self.log:
                log_string = f"{list(prior / self.k)};{list(zip_freq / np.sum(zip_freq))};"

            if self.deterministic:
                if self.log:
                    log_string += f"{list(posterior.mean.cpu().numpy())};"
                action = int(torch.argmax(posterior.mean))
            else:
                posterior = torch.distributions.Categorical(probs=posterior.sample())
                if self.log:
                    log_string += f"{list(posterior.probs.cpu().numpy())};"
                action = int(posterior.sample())

            if self.log:
                with open(f"logs/{self.env_name}/{self.current_time}/game_{self.episode_number}.log", "a+") as f:
                    f.write(log_string + f"{action};{self.k}\n")

            return action

    def _get_frequencies(self, acts, normalize=True):
        freq = np.zeros(shape=(self.num_actions,), dtype=np.float32)
        for a in acts:
            freq[a] += 1

        if normalize:
            return freq / self.k
        else:
            return freq

    def _get_weighted_frequencies(self, acts, distances, normalize=True):
        freq = np.zeros(shape=(self.num_actions,), dtype=np.float32)
        for a, d in zip(acts, distances):
            freq[a] += 1 / d

        if normalize:
            return freq / np.sum(freq)
        else:
            return freq

    def set_episode_number(self, n):
        self.episode_number = n
        if self.log:
            os.makedirs(f"logs/{self.env_name}/{self.current_time}", exist_ok=True)
            with open(f"logs/{self.env_name}/{self.current_time}/game_{self.episode_number}.log", "a+") as f:
                f.write("prior;zip_distrib;posterior;action;k\n")
