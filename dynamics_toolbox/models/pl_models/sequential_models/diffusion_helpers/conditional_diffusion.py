"""
Denoising networks for diffusion model

Author: Aravind Venugopal
"""

import torch
import torch.nn as nn
import numpy as np
from dynamics_toolbox.models.pl_models.sequential_models.diffusion_helpers.denoisers import Model_mlp_diff_embed


class Model_Cond_Diffusion(nn.Module):
    def __init__(self, input_dim: int, 
                      output_dim: int, n_hidden: int, embed_dim: int, x_dim: int, y_dim: int, beta1: float, beta2: float,
                      is_dropout: bool = False, is_batch: bool = True, activation: str = 'relu', net_type: str = 'transformer', use_prev: bool = False, 
                      n_T: int = 50, device: str = 'None', drop_prob: int=0.1, guide_w: int=0.0):
        super(Model_Cond_Diffusion, self).__init__()
        self.device = device
        self.beta1, self.beta2 = beta1, beta2
        
        self.nn_model = Model_mlp_diff_embed(x_dim = input_dim,
                                          n_hidden = n_hidden,
                                          y_dim = output_dim,
                                          embed_dim = embed_dim,
                                          output_dim=output_dim,
                                          is_dropout=is_dropout,
                                          is_batch=is_batch,
                                          activation=activation,
                                          net_type=net_type,
                                          use_prev=use_prev,
                                          )#.to(device) #do something about device

        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.guide_w = guide_w 
    '''
    def loss_on_batch(self, x_batch, y_batch):
        _ts = torch.randint(1, self.n_T + 1, (y_batch.shape[0], 1)).to(self.device)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[_ts] * noise

        # use nn model to predict noise
        noise_pred_batch = self.nn_model(y_t, x_batch, _ts / self.n_T, context_mask)

        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_pred_batch)
    '''

    def autoregress_sample(self, x_batch, ddpm_buffer, return_y_trace=False, extract_embedding=False):
        #x_batch shape will be (batch size, 1, x_dim)
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True 
        # how many noisy actions to begin with
        n_sample, seq_len = x_batch.shape[0], x_batch.shape[1]
        y_shape = (n_sample, seq_len, self._output_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(x_batch.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[1], x_batch.shape[0]).to(x_batch.device)
            context_mask[:, n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[1], x_batch.shape[0]).to(x_batch.device)


        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        return_y_trace = True
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(x_batch.device)
            t_is = t_is.repeat(n_sample, seq_len, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1, 1)
                t_is = t_is.repeat(2, 1, 1)

            z = torch.randn(y_shape).to(x_batch.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = ddpm_buffer["oneover_sqrta"].to(x_batch.device)[i] * (y_i - eps * ddpm_buffer["mab_over_sqrtmab"].to(x_batch.device)[i]) + ddpm_buffer["sqrt_beta_t"].to(x_batch.device)[i] * z
            if return_y_trace and (i % 20 == 0 or i == self._diff_model.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return {'y_i': y_i.squeeze(dim = 1), 'y_i_store': y_i_store}
        else:
            return {'y_i': y_i.squeeze(dim = 1)}

    '''
    def sample_update(self, x_batch, betas, n_T, return_y_trace=False):
        original_nT = self.n_T

        # set new schedule
        self.n_T = n_T
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # I'm a bit confused why we are adding noise during denoising?
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        # reset original schedule
        self.n_T = original_nT
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i '''