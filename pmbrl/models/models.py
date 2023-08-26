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
        self.in_size = in_size
        self.out_size = out_size
        self.ensemble_size = ensemble_size
        self.act_fn_name = act_fn
        self.act_fn = self._get_act_fn(self.act_fn_name)
        self.reset_parameters()

    def forward(self, x):
        op = torch.baddbmm(self.biases, x, self.weights)
        op = self.act_fn(op)
        return op

    def reset_parameters(self):
        weights = torch.zeros(self.ensemble_size, self.in_size, self.out_size).float()
        biases = torch.zeros(self.ensemble_size, 1, self.out_size).float()

        for weight in weights:
            self._init_weight(weight, self.act_fn_name)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def _init_weight(self, weight, act_fn_name):
        if act_fn_name == "swish":
            nn.init.xavier_uniform_(weight)
        elif act_fn_name == "linear":
            nn.init.xavier_normal_(weight)

    def _get_act_fn(self, act_fn_name):
        if act_fn_name == "swish":
            return swish
        elif act_fn_name == "linear":
            return lambda x: x


class EnsembleModel(nn.Module):
    def __init__(
            self,
            in_size,
            out_size,
            hidden_size,
            ensemble_size,
            normalizer,
            act_fn="swish",
            device="cpu",
    ):
        super().__init__()

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
        self.dropout = nn.Dropout(p=0.2)

        self.ensemble_size = ensemble_size
        self.normalizer = normalizer
        self.device = device
        self.max_logvar = -1
        self.min_logvar = -5
        self.device = device
        self.to(device)

    def forward(self, embedded, actions):
        next_embedded_mean, next_embedded_var = self._propagate_network(
            embedded, actions
        )
        # delta_mean, delta_var = self._post_process_model_outputs(
        #     norm_delta_mean, norm_delta_var
        # )
        return next_embedded_mean, next_embedded_var

    def loss(self, embedded, actions, next_embedded):
        index = torch.randperm(embedded.shape[0])
        inverse_index = torch.argsort(index)
        embedded = embedded[index]
        actions = actions[index]
        next_embedded = next_embedded[index]
        # states, actions = self._pre_process_model_inputs(embedded, actions)
        # delta_targets = self._pre_process_model_targets(embedded_delta)
        next_mu, next_var = self._propagate_network(embedded, actions)
        loss = (next_mu - next_embedded) ** 2 / next_var + 0.1 * torch.log(next_var)
        loss = loss.mean(-1).mean(-1).sum()
        next_mu = next_mu[inverse_index]
        next_var = next_var[inverse_index]
        next_embedded=Normal(next_mu, torch.sqrt(next_var)).rsample()
        return loss, next_embedded

    def multi_step_loss(self, embedded, actions):
        embedded_cut=embedded[:,:self.ensemble_size]
        actions_cut=actions[:,:self.ensemble_size]
        emebedded_predict_mu=torch.zeros(embedded.shape[0],embedded.shape[0],self.ensemble_size,embedded.shape[2]).to(self.device) # from to
        emebedded_predict_var=torch.zeros(embedded.shape[0],embedded.shape[0],self.ensemble_size,embedded.shape[2]).to(self.device)
        emebedded_predict=torch.zeros(embedded.shape[0],embedded.shape[0],self.ensemble_size,embedded.shape[2]).to(self.device)
        for i in range(embedded.shape[0]):
            emebedded_predict[i,i]=embedded_cut[i]
        for i in range(embedded.shape[0]):
            for j in range(i+1,embedded.shape[0]):
                next_mu, next_var = self._propagate_network(emebedded_predict[i,j-1].unsqueeze(1), actions_cut[j-1].unsqueeze(1))
                next_mu, next_var = next_mu.squeeze(),next_var.squeeze()
                emebedded_predict_mu[i,j]=next_mu
                emebedded_predict_var[i,j]=next_var
                emebedded_predict[i,j]=Normal(next_mu, torch.sqrt(next_var)).rsample()

        total_loss=0
        for i in range(embedded.shape[0]-1):
            loss=0
            for j in range(i+1,embedded.shape[0]):
                loss+=(emebedded_predict_mu[i,j]-embedded_cut[j])**2/emebedded_predict_var[i,j]+0.1*torch.log(emebedded_predict_var[i,j])
            total_loss+=loss.mean(-1).sum()/(embedded.shape[0]-1-i)
        total_loss/=embedded.shape[0]

        next_mu, next_var = self._propagate_network(embedded, actions)
        next_embedded = Normal(next_mu, torch.sqrt(next_var)).rsample()
        return total_loss, next_embedded

    def sample(self, mean, var):
        return Normal(mean, torch.sqrt(var)).sample()

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_4.reset_parameters()
        self.to(self.device)

    def _propagate_network(self, states, actions):
        inp = torch.cat((states, actions), dim=2)
        op = self.fc_1(inp)
        op = self.fc_2(op)
        op = self.dropout(op)
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
    def __init__(self, in_size, hidden_size, act_fn="relu", device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_fn = getattr(F, act_fn)

        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.to(device)

    def forward(self, embedding_states):
        inp = embedding_states
        reward = self.act_fn(self.dropout1(self.fc_1(inp)))
        reward = self.act_fn(self.dropout2(self.fc_2(reward)))
        reward = self.fc_3(reward).squeeze(dim=1)
        return reward

    def loss(self, states, rewards):
        # 加入L2正则项抵抗过拟合，失败，会导致奖励拟合的效果变差（任意输入，都相同输出）
        r_hat = self(states)
        return F.l1_loss(r_hat, rewards)

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()


class CriticModel(nn.Module):
    def __init__(self, state_size, actions_size, embedded_size, hidden_size, act_fn="relu", device="cuda"):
        super().__init__()
        self.in_size = state_size + actions_size + embedded_size
        self.hidden_size = hidden_size
        self.device = device
        self.act_fn = getattr(F, act_fn)

        self.fc_1 = nn.Linear(self.in_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_3 = nn.Linear(self.hidden_size, 1)
        self.dropout1 = nn.Dropout(p=0.2)

        self.to(device)

    def forward(self, vec_obs,action,next_embedding):
        inp = torch.cat((vec_obs, action, next_embedding), dim=2)
        value = self.act_fn(self.dropout1(self.fc_1(inp)))
        value = self.act_fn(self.fc_2(value))
        value = self.fc_3(value).squeeze(dim=2)
        return value

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
