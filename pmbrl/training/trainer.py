# pylint: disable=not-callable
# pylint: disable=no-member

import torch

CHUNK_LENGTH = 11
ENC_DEC_TRAIN_EPOCH = 50
VEC_RECON_SCALE = 10
IMG_RECON_SCALE = 1


class Trainer(object):
    def __init__(
            self,
            encoder,
            decoder,
            ensemble,
            reward_model,
            buffer,
            n_train_epochs,
            batch_size,
            learning_rate,
            epsilon,
            grad_clip_norm,
            logger=None,
            device="cuda"
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger
        self.device = device

        self.params = list(ensemble.parameters()) + list(reward_model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

        self.enc_dec_params = list(encoder.parameters()) + list(decoder.parameters())
        self.enc_dec_optim = torch.optim.Adam(self.enc_dec_params, lr=1e-4, eps=1e-4)

    def train(self):
        e_losses = []
        r_losses = []
        vec_recon_losses = []
        img_recon_losses = []
        n_batches = []
        for epoch in range(1, self.n_train_epochs + 1):
            e_losses.append([])
            r_losses.append([])
            vec_recon_losses.append([])
            img_recon_losses.append([])
            n_batches.append(0)

            vec_obs, img_obs, actions, rewards, done = self.buffer.sample(self.batch_size, chunk_length=CHUNK_LENGTH)
            self.ensemble.train()
            self.reward_model.train()
            self.encoder.train()
            self.decoder.train()

            vec_obs = torch.as_tensor(vec_obs, device=self.device).transpose(0, 1)
            img_obs = torch.as_tensor(img_obs, device=self.device).transpose(0, 1)
            actions = torch.as_tensor(actions, device=self.device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=self.device).transpose(0, 1)

            flatten_vec_obs=vec_obs.reshape(-1, 10)
            flatten_img_obs=img_obs.reshape(-1, 3, 64, 64)


            self.optim.zero_grad()
            self.enc_dec_optim.zero_grad()
            embedded_obs = self.encoder(flatten_vec_obs, flatten_img_obs)
            embedded_obs = embedded_obs.view(CHUNK_LENGTH, self.batch_size, 20)

            e_loss = self.ensemble.loss(embedded_obs[:-1, ],
                                        actions[:-1, ],
                                        embedded_obs[1:, ])
            r_loss = self.reward_model.loss(embedded_obs, actions, rewards)
            (e_loss + r_loss).backward()

            flatten_recon_vec_obs, flatten_recon_img_obs = self.decoder(embedded_obs)
            recon_vec_obs = flatten_recon_vec_obs.view(CHUNK_LENGTH, self.batch_size, 10)
            recon_img_obs = flatten_recon_img_obs.view(CHUNK_LENGTH, self.batch_size, 3, 64, 64)
            vec_loss = VEC_RECON_SCALE * torch.nn.functional.mse_loss(vec_obs, recon_vec_obs, reduction='none').\
                mean([0, 1]).sum()
            img_loss = IMG_RECON_SCALE * torch.nn.functional.mse_loss(img_obs, recon_img_obs, reduction='none').\
                mean([0, 1]).sum()
            recon_loss = vec_loss + img_loss
            recon_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.params, self.grad_clip_norm, norm_type=2
            )
            torch.nn.utils.clip_grad_norm_(
                self.enc_dec_params, self.grad_clip_norm, norm_type=2
            )
            self.enc_dec_optim.step()
            self.optim.step()

            e_losses[epoch - 1].append(e_loss.item())
            r_losses[epoch - 1].append(r_loss.item())
            vec_recon_losses[epoch - 1].append(vec_loss.item())
            img_recon_losses[epoch - 1].append(img_loss.item())
            n_batches[epoch - 1] += 1

            if self.logger is not None and epoch % 20 == 0:
                avg_e_loss = self._get_avg_loss(e_losses, n_batches, epoch)
                avg_r_loss = self._get_avg_loss(r_losses, n_batches, epoch)
                avg_vec_recon_loss = self._get_avg_loss(vec_recon_losses, n_batches, epoch)
                avg_img_recon_loss = self._get_avg_loss(img_recon_losses, n_batches, epoch)
                message = "> Train epoch {} [ensemble {:.2f} | reward {:.2f}] [vec_recon {:.2f} | img_recon {:.2f}]"
                self.logger.log(message.format(epoch, avg_e_loss, avg_r_loss, avg_vec_recon_loss, avg_img_recon_loss))

        return (
            self._get_avg_loss(e_losses, n_batches, epoch),
            self._get_avg_loss(r_losses, n_batches, epoch),
            self._get_avg_loss(vec_recon_losses, n_batches, epoch),
            self._get_avg_loss(img_recon_losses, n_batches, epoch),
        )

    def train_enc_dec(self):
        vec_recon_losses = []
        img_recon_losses = []
        n_batches = []
        for epoch in range(1, ENC_DEC_TRAIN_EPOCH + 1):
            vec_recon_losses.append([])
            img_recon_losses.append([])
            n_batches.append(0)
            vec_obs, img_obs, _, _, _ = self.buffer.sample(self.batch_size, chunk_length=CHUNK_LENGTH)
            self.encoder.train()
            self.decoder.train()

            vec_obs = torch.as_tensor(vec_obs, device=self.device).transpose(0, 1)  # horizon, batch, dim
            flatten_vec_obs = vec_obs.reshape(-1, 10)
            img_obs = torch.as_tensor(img_obs, device=self.device).transpose(0, 1)
            flatten_img_obs = img_obs.reshape(-1, 3, 64, 64)
            embedded_obs = self.encoder(flatten_vec_obs, flatten_img_obs)
            flatten_recon_vec_obs, flatten_recon_img_obs = self.decoder(embedded_obs)
            recon_vec_obs = flatten_recon_vec_obs.view(CHUNK_LENGTH,self.batch_size, 10)
            recon_img_obs = flatten_recon_img_obs.view(CHUNK_LENGTH,self.batch_size, 3, 64, 64)
            self.optim.zero_grad()
            vec_loss = VEC_RECON_SCALE * torch.nn.functional.mse_loss(vec_obs, recon_vec_obs).mean([0,1]).sum()
            img_loss = IMG_RECON_SCALE * torch.nn.functional.mse_loss(img_obs, recon_img_obs,reduction='none').mean([0,1]).sum()
            recon_loss = vec_loss + img_loss
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.enc_dec_params, self.grad_clip_norm, norm_type=2
            )
            self.enc_dec_optim.step()

            vec_recon_losses[epoch - 1].append(vec_loss.item())
            img_recon_losses[epoch - 1].append(img_loss.item())
            n_batches[epoch - 1] += 1

            if self.logger is not None and epoch % 10 == 0:
                avg_vec_loss = self._get_avg_loss(vec_recon_losses, n_batches, epoch)
                avg_img_loss = self._get_avg_loss(img_recon_losses, n_batches, epoch)
                message = "> Train epoch {} [vec_loss {:.2f} | img_loss {:.2f}]"
                self.logger.log(message.format(epoch, avg_vec_loss, avg_img_loss))

        return (
            self._get_avg_loss(vec_recon_losses, n_batches, epoch),
            self._get_avg_loss(img_recon_losses, n_batches, epoch),
        )

    def reset_models(self):
        self.ensemble.reset_parameters()
        self.reward_model.reset_parameters()
        self.params = list(self.ensemble.parameters()) + list(
            self.reward_model.parameters()
        )
        self.optim = torch.optim.Adam(
            self.params, lr=self.learning_rate, eps=self.epsilon
        )

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batch for loss, n_batch in zip(losses, n_batches)]
        return sum(epoch_loss) / epoch
