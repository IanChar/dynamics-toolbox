"""
Wrapper for joint_horizon_distribution modeling for trajectory generation.

Author: Youngseog Chung
Date: May 8, 2023
"""

import os
from typing import Dict

import tqdm
import numpy as np
import torch


def batch_conditional_sampling_with_joint_correlation(
        corr_mat: torch.Tensor,
        hist_obs: torch.Tensor,
        hist_means: torch.Tensor,
        hist_stds: torch.Tensor,
        pred_means: torch.Tensor,
        pred_stds: torch.Tensor,
        num_draw_per_rv: int = 1,
):
    """

    Args:
        corr_mat: correlation matrix across horizon, (H, H, obs_dim)
        hist_obs: observations so far, (num_samples, num_hist_obs, obs_dim)
            assume the observations always start from H=0
        hist_means: means of observations so far, (num_samples, num_hist_obs, obs_dim)
        hist_stds: stds of observations so far, (num_samples, num_hist_obs, obs_dim)
        pred_means: predicted means for future observations, (num_samples, num_pred, obs_dim)
        pred_stds: predicted stds for future observations, (num_samples, num_pred, obs_dim)
        num_draw_per_rv: number of samples to draw for each prediction

    Returns:

    """
    H, _, obs_dim = corr_mat.shape  # H is full horizon
    num_samples, num_hist, obs_dim = hist_obs.shape
    num_pred = pred_means.shape[1]
    assert corr_mat.shape == (H, H, obs_dim)
    assert (hist_obs.shape == hist_means.shape == hist_stds.shape
            == (num_samples, num_hist, obs_dim))
    assert pred_means.shape == pred_stds.shape == (num_samples, num_pred, obs_dim)
    # corr_mat = np.tile(corr_mat, (num_samples, 1, 1, 1))  # (num_samples, H, H, obs_dim)
    corr_mat = corr_mat.repeat(num_samples, 1, 1, 1)  # (num_samples, H, H, obs_dim)
    assert corr_mat.shape == (num_samples, H, H, obs_dim)
    # print(f"Horizon: {H}, num_hist: {num_hist}, num_pred: {num_pred}, num_samples: {num_samples}")

    hist_block_cov = (hist_stds.unsqueeze(1)
                      * corr_mat[:, :num_hist, :num_hist, :]
                      * hist_stds.unsqueeze(2))
    pred_hist_cov = (pred_stds.unsqueeze(2)
                     * corr_mat[:, num_hist:(num_hist + num_pred), :num_hist, :]
                     * hist_stds.unsqueeze(1))
    hist_pred_cov = (hist_stds.unsqueeze(2)
                     * corr_mat[:, :num_hist, num_hist:(num_hist + num_pred), :]
                     * pred_stds.unsqueeze(1))
    pred_pred_cov = (
            pred_stds.unsqueeze(1)
            * corr_mat[:, num_hist:(num_hist + num_pred),
              num_hist:(num_hist + num_pred), :]
            * pred_stds.unsqueeze(2))

    torch.allclose(hist_pred_cov, pred_hist_cov.permute(0, 2, 1, 3))
    assert hist_block_cov.shape == (num_samples, num_hist, num_hist, obs_dim)
    assert pred_hist_cov.shape == (num_samples, num_pred, num_hist, obs_dim)
    assert hist_pred_cov.shape == (num_samples, num_hist, num_pred, obs_dim)
    assert pred_pred_cov.shape == (num_samples, num_pred, num_pred, obs_dim)

    hist_block_cov = hist_block_cov.permute(0, 3, 1,
                                              2)  # (num_samples, obs_dim, num_hist, num_hist)
    pred_hist_cov = pred_hist_cov.permute(0, 3, 1,
                                            2)  # (num_samples, obs_dim, num_pred, num_hist)
    hist_pred_cov = hist_pred_cov.permute(0, 3, 1,
                                            2)  # (num_samples, obs_dim, num_hist, num_pred)
    pred_pred_cov = pred_pred_cov.permute(0, 3, 1,
                                            2)  # (num_samples, obs_dim, num_pred, num_pred)
    hist_obs = hist_obs.permute(0, 2, 1)  # (num_samples, obs_dim, num_hist)
    hist_means = hist_means.permute(0, 2, 1)  # (num_samples, obs_dim, num_hist)
    pred_means = pred_means.permute(0, 2, 1)  # (num_samples, obs_dim, num_pred)

    assert hist_block_cov.shape == (num_samples, obs_dim, num_hist, num_hist)
    assert pred_hist_cov.shape == (num_samples, obs_dim, num_pred, num_hist)
    assert hist_pred_cov.shape == (num_samples, obs_dim, num_hist, num_pred)
    assert pred_pred_cov.shape == (num_samples, obs_dim, num_pred, num_pred)
    assert hist_obs.shape == (num_samples, obs_dim, num_hist)
    assert hist_means.shape == (num_samples, obs_dim, num_hist)
    assert pred_means.shape == (num_samples, obs_dim, num_pred)

    # cond_mean = pred_means + np.matmul(
    #     # (num_samples, obs_dim, num_pred, num_hist) x (num_samples, obs_dim, num_hist, num_hist) => (num_samples, obs_dim, num_pred, num_hist)
    #     np.matmul(pred_hist_cov, np.linalg.inv(hist_block_cov)),
    #     # (num_samples, obs_dim, num_hist) - (num_samples, obs_dim, num_hist) => (num_samples, obs_dim, num_hist)
    #     # => (num_samples, obs_dim, num_hist, 1)
    #     (hist_obs - hist_means)[..., np.newaxis])[..., 0]
    cond_mean = pred_means + torch.matmul(
        # (num_samples, obs_dim, num_pred, num_hist) x (num_samples, obs_dim, num_hist, num_hist) => (num_samples, obs_dim, num_pred, num_hist)
        torch.matmul(pred_hist_cov, torch.inverse(hist_block_cov)),
        # (num_samples, obs_dim, num_hist) - (num_samples, obs_dim, num_hist) => (num_samples, obs_dim, num_hist)
        # => (num_samples, obs_dim, num_hist, 1)
        (hist_obs - hist_means).unsqueeze(-1))[..., 0]
    # outer matmul: (num_samples, obs_dim, num_pred, num_hist) x (num_samples, obs_dim, num_hist, 1)
    # => (num_samples, obs_dim, num_pred, 1), then index [..., 0] => (num_samples, obs_dim, num_pred)
    assert cond_mean.shape == (num_samples, obs_dim, num_pred)

    # cond_cov = pred_pred_cov - np.matmul(
    #     # (num_samples, obs_dim, num_pred, num_hist) x (num_samples, obs_dim, num_hist, num_hist) => (num_samples, obs_dim, num_pred, num_hist)
    #     np.matmul(pred_hist_cov, np.linalg.inv(hist_block_cov)),
    #     # (num_samples, obs_dim, num_hist, num_pred)
    #     hist_pred_cov)
    cond_cov = pred_pred_cov - torch.matmul(
        # (num_samples, obs_dim, num_pred, num_hist) x (num_samples, obs_dim, num_hist, num_hist) => (num_samples, obs_dim, num_pred, num_hist)
        torch.matmul(pred_hist_cov, torch.inverse(hist_block_cov)),
        # (num_samples, obs_dim, num_hist, num_pred)
        hist_pred_cov)
    # outer matmul: (num_samples, obs_dim, num_pred, num_hist) x (num_samples, obs_dim, num_hist, num_pred)
    # => (num_samples, obs_dim, num_pred, num_pred)
    assert cond_cov.shape == (num_samples, obs_dim, num_pred, num_pred)

    cond_samples_np = np.zeros((num_samples, obs_dim, num_pred, num_draw_per_rv))
    # cond_samples = torch.zeros((num_samples, obs_dim, num_pred, num_draw_per_rv))

    ### BEGIN: new code
    # breakpoint()
    # for b in range(num_samples):
    mean_list_per_dim = []
    cov_list_per_dim = []
    cond_samples_per_dim = []
    for i in range(obs_dim):
        # dim_samples = np.random.multivariate_normal(cond_mean[b, i], cond_cov[b, i],
        #                                             size=num_draw_per_rv).T  # (num_pred, num_draw_per_rv)
        dim_distrs = torch.distributions.MultivariateNormal(cond_mean[:, i, :], cond_cov[:, i, :, :])
        dim_samples = dim_distrs.sample((num_draw_per_rv,)).permute(1, 2, 0)  
        assert dim_samples.shape == (num_samples, num_pred, num_draw_per_rv)
        # cond_samples[:, i] = dim_samples
        cond_samples_per_dim.append(dim_samples)
        mean_list_per_dim.append(cond_mean[:, i, :])
        cov_list_per_dim.append(cond_cov[:, i, :, :])
    cond_samples = torch.stack(cond_samples_per_dim, dim=1)
    mean_list = torch.stack(mean_list_per_dim, dim=-1)  # (num_pred, obs_dim)
    cov_list = torch.stack(cov_list_per_dim, dim=-1)  # (num_pred, num_pred, obs_dim)
    ###
    # ###
    # print('old_code')
    # mean_list = []
    # cov_list = []
    # # I have to iterate over sample_idx and obs_idx because numpy's multivariate_normal only supports 2D inputs
    # for b in tqdm.tqdm(range(num_samples)):
    #     mean_list_per_sample = []
    #     cov_list_per_sample = []
    #     for i in range(obs_dim):
    #         dim_samples = np.random.multivariate_normal(cond_mean[b, i].cpu().numpy(), cond_cov[b, i].cpu().numpy(),
    #                                                     size=num_draw_per_rv).T  # (num_pred, num_draw_per_rv)
    #         cond_samples_np[b, i] = dim_samples
    #         mean_list_per_sample.append(cond_mean[b, i].cpu().numpy())
    #         cov_list_per_sample.append(cond_cov[b, i].cpu().numpy())
    #     mean_list.append(np.stack(mean_list_per_sample, axis=-1))  # (num_pred, obs_dim)
    #     cov_list.append(
    #         np.stack(cov_list_per_sample, axis=-1))  # (num_pred, num_pred, obs_dim)
    # mean_list = np.stack(mean_list, axis=0)  # (num_samples, num_pred, obs_dim)
    # cov_list = np.stack(cov_list, axis=0)  # (num_samples, num_pred, num_pred, obs_dim)
    # ###

    # breakpoint()
    # check_mean = torch_mean_list.cpu().numpy()
    # check_cov = torch_cov_list.cpu().numpy()




    out_samples = cond_samples.permute(0, 2, 3,
                                         1)  # (num_samples, num_pred, num_draw_per_rv, obs_dim)
    out_means = mean_list
    out_covs = cov_list
    # out_means = np.stack(mean_list, axis=0)
    # out_covs = np.stack(cov_list, axis=0)

    assert out_means.shape == (num_samples, num_pred, obs_dim)
    assert out_covs.shape == (num_samples, num_pred, num_pred, obs_dim)
    assert out_samples.shape == (num_samples, num_pred, num_draw_per_rv, obs_dim)

    out_info = {'means': out_means, 'covs': out_covs}

    return out_samples, out_info


class joint_horizon_distribution_wrapper():
    def __init__(self, wrapped_model, error_corr_mat_path, recal_constants_path=None, do_not_apply_corr=False):

        # attributes to get from wrapped_model
        self.wrapped_model = wrapped_model
        self.input_dim = wrapped_model.input_dim
        self.output_dim = wrapped_model.output_dim
        self.model_device = None
        self.updated_model_device = False

        # check the dimensions of error_corr_mat
        self.error_corr_mat = torch.from_numpy(np.load(error_corr_mat_path))
        self.do_not_apply_corr = do_not_apply_corr
        ### temp code for testing
        if len(self.error_corr_mat.shape) == 2:
            dim_1, dim_2 = self.error_corr_mat.shape
            assert dim_1 == dim_2, "error_corr_mat must be (horizon, horizon) or (horizon, horizon, dim)"
            self.error_corr_mat = np.tile(self.error_corr_mat[..., None], (1, 1, self.output_dim))
            print(self.error_corr_mat.shape)
        ###
        assert (len(self.error_corr_mat.shape) == 3,
                "error_corr_mat must be (horizon, horizon, dim)")
        assert self.error_corr_mat.shape[0] == self.error_corr_mat.shape[
            1], "error_corr_mat must be (horizon, horizon, dim)"
        self.max_horizon, _, self.obs_dim = self.error_corr_mat.shape
        assert self.obs_dim == self.output_dim, "output_dim of model must match obs_dim"

        # attributes for generation
        self.h_idx = 0
        # h_idx is horizon index to keep track of the current prediction timestep
        # -> i.e. which row-column index of error_corr_mat to use
        # => use h_idx of error_corr_mat to generate pred for time=(h_idx+1)
        self.observed_gamma = []

        # flags to set later in case recalibration is applied
        self.recal_constants = None
        self.apply_recal = False
        if recal_constants_path is not None:
            recal_constants = torch.from_numpy(np.load(recal_constants_path))
            self.set_recalibration(recal_constants)
        # NOTE: apply self.recal_constants[h_idx] to predictions made for time=(h_idx+1)
        # -> apply recal_constants[h_idx] when predicting AT time=(h_idx)

    # def set_wrapped_model(self, model):
    #     self.wrapped_model = model
    #     self.input_dim = model.input_dim
    #     self.output_dim = model.output_dim
    #     assert self.output_dim == self.obs_dim, "output_dim of model must match obs_dim"

    def set_recalibration(self, recal_constants):
        # recal_constants should be of shape (horizon, output_dim)
        self.apply_recal = True
        self.recal_constants = recal_constants

    def reset(self):
        # need to reset self.h_idx to 0
        self.h_idx = 0
        self.observed_gamma = []

    def _handle_mixture_model(self, info: Dict[str, torch.Tensor]):
        """If we get multiple predictions assume we are dealing with a Gaussian
           mixture model and handle accordingly.
        """
        if len(info['mean_predictions'].shape) == 2:
            if isinstance(info['mean_predictions'], np.ndarray):
                return (torch.from_numpy(info['mean_predictions']).to(self.model_device),
                        torch.from_numpy(info['std_predictions']).to(self.model_device))
            return (info['mean_predictions'].to(self.model_device),
                    info['std_predictions'].to(self.model_device))
        if isinstance(info['mean_predictions'], np.ndarray):
            means, stds = (torch.from_numpy(info['mean_predictions']).to(self.model_device),
                           torch.from_numpy(info['std_predictions']).to(self.model_device))
        else:
            means, stds = (info['mean_predictions'].to(self.model_device),
                           info['std_predictions'].to(self.model_device))
        members = len(means)
        mean_out = torch.mean(means, dim=0)
        mean_var = torch.mean(stds.pow(2), dim=0)
        mean_sq = torch.mean(means.pow(2), dim=0) * (1 - 1 / members)
        mixing_term = 2 / (members ** 2) * torch.sum(torch.cat([torch.stack([
            means[i] * means[j]
            for j in range(i)])
            for i in range(1, members)], dim=0), dim=0)
        std_out = torch.sqrt(mean_var + mean_sq - mixing_term)
        return mean_out, std_out

    def predict(self, model_input, **kwargs):
        if not self.updated_model_device:
            self.model_device = getattr(self.wrapped_model.normalizer, '1_scaling').device
            self.error_corr_mat = self.error_corr_mat.to(self.model_device)
            if self.apply_recal:
                self.recal_constants = self.recal_constants.to(self.model_device)
            self.updated_model_device = True

        assert self.wrapped_model is not None, "self.wrapped_model must be set before calling predict"

        # get shapes from model_input
        batch_size, input_dim = model_input.shape
        assert input_dim == self.input_dim, "input_dim of model_input must match self.input_dim"

        # check attributes for generation
        assert self.h_idx < self.max_horizon, "h_idx must be less than max_horizon"
        assert self.h_idx == len(self.observed_gamma)

        # ### BEGIN: original code block
        # _, pred_info = self.wrapped_model.predict(model_input, **kwargs)
        # # get UNNORMALIZED mean and std
        # obs_delta_mean_preds = (self.wrapped_model._unnormalize_prediction_output(
        #     torch.from_numpy(pred_info['mean_predictions']))).numpy()
        # std_preds = (torch.from_numpy(pred_info['std_predictions'])
        #              * getattr(self.wrapped_model.normalizer, '1_scaling')).numpy()
        # ### END: original code block

        ### BEGIN: new code block
        temp_pred, pred_info = self.wrapped_model.predict(model_input, **kwargs)
        if self.do_not_apply_corr:
            return temp_pred, pred_info

        obs_delta_mean_preds, std_preds = self._handle_mixture_model(pred_info)
        # these are actually NORMALIZED mean and std
        # both torch tensors and on device
        ### END: new code block

        if self.h_idx == 0:
            # gamma = torch.from_numpy(np.random.normal(size=(batch_size, self.obs_dim)))
            gamma = torch.randn(batch_size, self.obs_dim, device=self.model_device)
        else:
            # hist_obs = np.stack(self.observed_gamma, axis=1)
            hist_obs = torch.stack(self.observed_gamma, dim=1).to(self.model_device)
            # should be (batch_size, h_idx, obs_dim)
            batch_size = hist_obs.shape[0]
            assert hist_obs.shape[1:] == (self.h_idx, self.obs_dim)

            gamma, _ = batch_conditional_sampling_with_joint_correlation(
                corr_mat=self.error_corr_mat[1:, 1:].float(),
                hist_obs=hist_obs.float(),
                hist_means=torch.zeros((batch_size, self.h_idx, self.obs_dim), device=self.model_device).float(),
                hist_stds=torch.ones((batch_size, self.h_idx, self.obs_dim), device=self.model_device).float(),
                pred_means=torch.zeros((batch_size, 1, self.obs_dim), device=self.model_device).float(),
                pred_stds=torch.ones((batch_size, 1, self.obs_dim), device=self.model_device).float(),
                num_draw_per_rv=1
            )
            gamma = gamma[:, 0, 0, :]
        assert obs_delta_mean_preds.shape == gamma.shape == std_preds.shape, "shapes of obs_delta_mean_preds, gamma, and std_preds must match"
        if self.apply_recal:
            use_std = std_preds * self.recal_constants[self.h_idx]
        else:
            use_std = std_preds

        obs_delta = (obs_delta_mean_preds + gamma * use_std).to(self.model_device).float()
        ### BEGIN: new code block
        obs_delta = self.wrapped_model._unnormalize_prediction_output(obs_delta).cpu().numpy()
        ### END: new code block
        info = {'mean_predictions': obs_delta_mean_preds.cpu().numpy(),  # this is non-normalized
                'std_predictions': (gamma * use_std).cpu().numpy()}  # this is non-normalized

        self.observed_gamma.append(gamma)
        self.h_idx += 1

        return obs_delta, info
