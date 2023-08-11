# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import numpy as np
import datetime


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

        self.vec_obs = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.img_obs = np.zeros((buffer_size, 3, 64, 64), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done = np.zeros((buffer_size, 1), dtype=np.bool_)

        # self.normalizer = normalizer
        self._total_steps = 0

    def add(self, state, action, reward, done):
        vec, img = state['vec'], state['img']
        img = np.traspose(img, (2, 0, 1))
        idx = self._total_steps % self.buffer_size

        self.vec_obs[idx] = vec
        self.img_obs[idx] = img
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.done[idx] = done
        self._total_steps += 1

        # self.normalizer.update(state, action, vec_delta)
        # TODO normalizer

    def save(self, dir_path='tactile_push_run'):
        now = datetime.datetime.now()
        path = dir_path + f'/replay_buffer_{now.month:>2}{now.day:>2}_{now.hour:>2}{now.minute:>2}.npz'
        np.savez_compressed(path,
                            vec_obs=self.vec_obs,
                            img_obs=self.img_obs,
                            actions=self.actions,
                            rewards=self.rewards,
                            done=self.done,
                            _total_steps=self._total_steps)
        print(f"saving buffer to {path}, length with {self._total_steps}")

    def load(self, dir_path='tactile_push_run', file='replay_buffer.npz'):
        path= dir_path + file
        print(f"loading replay buffer from {path}")
        data = np.load(path)
        self.vec_obs = data['vec_obs']
        self.img_obs = data['img_obs']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.done = data['done']
        self._total_steps = data['_total_steps']
        print("loading replay buffer finish with index : ", self._total_steps)

    def sample(self, batch_size, chunk_length):
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1)
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_vec_observations = self.vec_obs[sampled_indexes].reshape(
            batch_size, chunk_length, *self.vec_obs.shape[1:])
        sampled_img_observations = self.img_obs[sampled_indexes].reshape(
            batch_size, chunk_length, *self.img_obs.shape[1:])

        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_vec_observations, sampled_img_observations, sampled_actions, sampled_rewards, sampled_done

    # def get_train_batches(self, batch_size):
    #     size = len(self)
    #     indices = [
    #         np.random.permutation(range(size)) for _ in range(self.ensemble_size)
    #     ]
    #     indices = np.stack(indices).T
    #
    #     for i in range(0, size, batch_size):
    #         j = min(size, i + batch_size)
    #
    #         if (j - i) < batch_size and i != 0:
    #             return
    #
    #         batch_size = j - i
    #
    #         batch_indices = indices[i:j]
    #         batch_indices = batch_indices.flatten()
    #
    #         states = self.states[batch_indices]
    #         pic_obs = self.pic_obs[batch_indices]
    #         actions = self.actions[batch_indices]
    #         rewards = self.rewards[batch_indices]
    #         state_deltas = self.state_deltas[batch_indices]
    #         next_pic_obs = self.next_pic_obs[batch_indices]
    #
    #         states = torch.from_numpy(states).float().to(self.device)
    #         pic_obs = torch.from_numpy(pic_obs).float().to(self.device)
    #         actions = torch.from_numpy(actions).float().to(self.device)
    #         rewards = torch.from_numpy(rewards).float().to(self.device)
    #         state_deltas = torch.from_numpy(state_deltas).float().to(self.device)
    #         next_pic_obs = torch.from_numpy(next_pic_obs).float().to(self.device)
    #
    #         if self.signal_noise is not None:
    #             states = states + self.signal_noise * torch.randn_like(states)
    #
    #         states = states.reshape(self.ensemble_size, batch_size, self.state_size)
    #         pic_obs = pic_obs.reshape(self.ensemble_size, batch_size, 3, 64, 64)
    #         actions = actions.reshape(self.ensemble_size, batch_size, self.action_size)
    #         rewards = rewards.reshape(self.ensemble_size, batch_size, 1)
    #         state_deltas = state_deltas.reshape(
    #             self.ensemble_size, batch_size, self.state_size
    #         )
    #         next_pic_obs = next_pic_obs.reshape(self.ensemble_size, batch_size, 3, 64, 64)
    #
    #         yield states, pic_obs, actions, rewards, state_deltas, next_pic_obs

    def __len__(self):
        return min(self._total_steps, self.buffer_size)

    @property
    def total_steps(self):
        return self._total_steps
