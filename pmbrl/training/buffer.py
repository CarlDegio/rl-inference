# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np


class Buffer(object):
    def __init__(
            self,
            state_size,
            action_size,
            ensemble_size,
            normalizer,
            signal_noise=None,
            buffer_size=10 ** 5,
            device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.ensemble_size = ensemble_size
        self.buffer_size = buffer_size
        self.signal_noise = signal_noise
        self.device = device

        self.states = np.zeros((buffer_size, state_size))
        self.actions = np.zeros((buffer_size, action_size))
        self.rewards = np.zeros((buffer_size, 1))
        self.state_deltas = np.zeros((buffer_size, state_size))

        self.pic_obs = np.zeros((buffer_size, 3, 64, 64))

        self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, next_state):
        vec, img = state['vec'], state['img']
        img=np.traspose(img, (2, 0, 1))
        idx = self._total_steps % self.buffer_size
        vec_delta= next_state['vec'] - vec

        self.states[idx] = vec
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.state_deltas[idx] = vec_delta
        self.pic_obs[idx] = img
        self._total_steps += 1

        self.normalizer.update(state, action, vec_delta)

    def save(self, path="buffer.npz"):
        length = min(self._total_steps, self.buffer_size)
        save_dict = {
            "length": length,
            "states": self.states[:length],
            "actions": self.actions[:length],
            "rewards": self.rewards[:length],
            "state_deltas": self.state_deltas[:length]
        }
        np.savez(path, **save_dict)
        print(f"saving buffer to {path}, length with {length}")

    def load(self, path="buffer.npz"):
        load_dict = np.load(path)
        print(f"loading buffer from {path}, length with {load_dict['length']}")
        for i in range(load_dict["length"]):
            self.add(load_dict["states"][i],
                     load_dict["actions"][i],
                     load_dict["rewards"][i],
                     load_dict["states"][i] + load_dict["state_deltas"][i])

    def get_train_batches(self, batch_size):
        size = len(self)
        indices = [
            np.random.permutation(range(size)) for _ in range(self.ensemble_size)
        ]
        indices = np.stack(indices).T

        for i in range(0, size, batch_size):
            j = min(size, i + batch_size)

            if (j - i) < batch_size and i != 0:
                return

            batch_size = j - i

            batch_indices = indices[i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            rewards = self.rewards[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = torch.from_numpy(states).float().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            state_deltas = torch.from_numpy(state_deltas).float().to(self.device)

            if self.signal_noise is not None:
                states = states + self.signal_noise * torch.randn_like(states)

            states = states.reshape(self.ensemble_size, batch_size, self.state_size)
            actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
            rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
            state_deltas = state_deltas.reshape(
                self.ensemble_size, batch_size, self.state_size
            )

            yield states, actions, rewards, state_deltas

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps
