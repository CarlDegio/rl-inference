# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def swish(x):
    return x * torch.sigmoid(x)


class EnsembleDenseLayer(nn.Module):
    def __init__(self, in_size, out_size, ensemble_size, act_fn="swish"):
        super().__init__()
        weights = torch.zeros(ensemble_size, in_size, out_size).float()
        biases = torch.zeros(ensemble_size, 1, out_size).float()

        for weight in weights:
            self._init_weight(weight, act_fn)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)
        self.act_fn = self._get_act_fn(act_fn)

    def forward(self, x):
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.act_fn(op)
        return op

    def _init_weight(self, weight, act_fn):
        if act_fn == "swish":
            nn.init.xavier_uniform_(weight)
        elif act_fn == "relu":
            nn.init.kaiming_normal_(weight)
        elif act_fn == "linear":
            nn.init.xavier_normal_(weight)

    def _get_act_fn(self, act_fn):
        if act_fn == "swish":
            return swish
        elif act_fn == "relu":
            return F.relu
        elif act_fn == "linear":
            return lambda x: x


class EnsembleModel(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        ensemble_size,
        normalizer,
        max_logvar=-1,
        min_logvar=-5,
        act_fn="swish",
        device="cpu",
    ):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.fc_1 = EnsembleDenseLayer(
            in_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_2 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_3 = EnsembleDenseLayer(
            hidden_size, hidden_size, ensemble_size, act_fn=act_fn
        )
        self.fc_4 = EnsembleDenseLayer(
            hidden_size, out_size * 2, ensemble_size, act_fn="linear"
        )

        self.normalizer = normalizer
        self.device = device

        self.max_logvar = max_logvar
        self.min_logvar = min_logvar
        self.to(device)

    def forward(self, states, actions):
        norm_states, norm_actions = self._pre_process_model_inputs(states, actions)
        norm_delta_mean, norm_delta_var = self._propagate_network(
            norm_states, norm_actions
        )
        delta_mean, delta_var = self._post_process_model_outputs(
            norm_delta_mean, norm_delta_var
        )
        return delta_mean, delta_var

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def loss(self, states, actions, state_deltas):
        states, actions = self._pre_process_model_inputs(states, actions)
        delta_targets = self._pre_process_model_targets(state_deltas)
        delta_mu, delta_var = self._propagate_network(states, actions)
        loss = (delta_mu - delta_targets) ** 2 / delta_var + torch.log(delta_var)
        loss = loss.mean(-1).mean(-1).sum()
        return loss

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.fc_1(inp)
        op = self.fc_2(op)
        op = self.fc_3(op)
        op = self.fc_4(op)

        delta_mean, delta_logvar = torch.split(op, op.size(2) // 2, dim=2)
        delta_logvar = torch.sigmoid(delta_logvar)
        delta_logvar = (
            self.min_logvar + (self.max_logvar - self.min_logvar) * delta_logvar
        )
        delta_var = torch.exp(delta_logvar)

        return delta_mean, delta_var

    def _pre_process_model_inputs(self, states, actions):
        states = states.to(self.device)
        actions = actions.to(self.device)
        states = self.normalizer.normalize_states(states)
        actions = self.normalizer.normalize_actions(actions)
        return states, actions

    def _pre_process_model_targets(self, state_deltas):
        state_deltas = state_deltas.to(self.device)
        state_deltas = self.normalizer.normalize_state_deltas(state_deltas)
        return state_deltas

    def _post_process_model_outputs(self, delta_mean, delta_var):
        delta_mean = self.normalizer.denormalize_state_delta_means(delta_mean)
        delta_var = self.normalizer.denormalize_state_delta_vars(delta_var)
        return delta_mean, delta_var


class RewardModel(nn.Module):
    def __init__(
        self,
        state_size,
        hidden_size,
        use_reward_ensemble=False,
        ensemble_size=None,
        act_fn="relu",
        device="cpu",
    ):
        super().__init__()
        self.use_reward_ensemble = use_reward_ensemble
        if not self.use_reward_ensemble:
            self.act_fn = getattr(F, act_fn)
            self.fc_1 = nn.Linear(state_size, hidden_size)
            self.fc_2 = nn.Linear(hidden_size, hidden_size)
            self.fc_3 = nn.Linear(hidden_size, 1)
        else:
            self.fc_1 = EnsembleDenseLayer(
                state_size, hidden_size, ensemble_size, act_fn=act_fn
            )
            self.fc_2 = EnsembleDenseLayer(
                hidden_size, hidden_size, ensemble_size, act_fn=act_fn
            )
            self.fc_3 = EnsembleDenseLayer(
                hidden_size, 1, ensemble_size, act_fn="linear"
            )
        self.to(device)

    def forward(self, state):
        if not self.use_reward_ensemble:
            reward = self.act_fn(self.fc_1(state))
            reward = self.act_fn(self.fc_2(reward))
            reward = self.fc_3(reward).squeeze(dim=1)
        else:
            reward = self.fc_1(state)
            reward = self.fc_2(reward)
            reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, rewards):
        r_hat = self(states)
        if not self.use_reward_ensemble:
            loss = F.mse_loss(r_hat, rewards)
        else:
            loss = F.mse_loss(r_hat, rewards, reduction="none")
            loss = loss.mean(-1).mean(-1).sum()
        return loss

