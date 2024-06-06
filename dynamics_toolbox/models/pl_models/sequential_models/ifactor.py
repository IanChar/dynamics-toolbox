"""
Recurssive world model that learns actionable and non-actionable states.
(https://arxiv.org/abs/2306.06561)

Author: Namrata Deka
"""
from cProfile import label
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import numpy as np
import hydra.utils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from torch.distributions import Normal, kl_divergence

from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
    import AbstractSequentialModel

from dynamics_toolbox.constants import sampling_modes

from dynamics_toolbox.utils.misc import get_parameters
from dynamics_toolbox.utils.pytorch.modules.mine import MINE
from dynamics_toolbox.utils.pytorch.metrics import SequentialExplainedVariance


class IFactor(AbstractSequentialModel):
    """IFactor network."""
    def __init__(self, 
        obs_dim: int,
        action_dim: int,
        sa_dim: int,
        sac_dim: int,
        obs_encoder_cfg: DictConfig,
        obs_decoder_cfg: DictConfig,
        sa_decoder_cfg: DictConfig,
        sac_decoder_cfg: DictConfig,
        action_ft_dim: Optional[int] = None,
        action_encoder_cfg: Optional[DictConfig] = None,
        rnn_num_layers: int = 1,
        rnn_type: str = 'gru',
        mi_steps: int = 3,
        mine_alpha: float = 0.01,
        mine_hidden_size: int = 128,
        labels_are_velocities: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: Optional[float] = 0.0,
        grad_clip_norm: Optional[float] = 100.0,
        obs_weight: float = 1.0,
        kl_weight: float = 1.0,
        mi_weight: float = 1.0,
        **kwargs
    ):
        """
        Constructor.

        Args:
            obs_dim: observed state space dimension
            action: action space dimension
            sa_dim: actionable latent state space dimension
            sac_dim: non-actionable latent state space dimension
            obs_encoder_cfg: configuration for the observation encoder network.
            obs_decoder_cfg: configuration for the observation decoder network.
            sa_decoder_cfg: configuration for the actionable latent predictor.
            sac_decoder_cfg: configuration for the non-actionable latent predictor.
            rnn_num_layers: Number of layers in the memory units.
            rnn_type: Name of the rnn type. Can accept GRU or LSTM.
            mi_steps: number of training steps for the MINE model.
            mine_alpha:
            mine_hidden_size: number of hidden units in MINE layers.
            
        """
        self._labels_are_velocities = labels_are_velocities
        self._input_dim = obs_dim + action_dim
        self._output_dim = obs_dim
        kwargs.update({'automatic_optimization': False})
        super().__init__(**kwargs)
        # self.automatic_optimization = False
        self.sa_dim, self.sac_dim = sa_dim, sac_dim
        state_dim = sa_dim + sac_dim
        # action encoder (MLP):
        if action_encoder_cfg is not None:
            self.action_encoder = hydra.utils.instantiate(
                action_encoder_cfg,
                input_dim = action_dim,
                output_dim = action_ft_dim,
                _recursive_=False,
            )
        else:
            self.action_encoder = None
        # observation encoder (PNN): o_t -> s_t
        self.obs_encoder = hydra.utils.instantiate(
            obs_encoder_cfg,
            input_dim = obs_dim,
            output_dim = state_dim,
            _recursive_ = False,
        )
        # observation decoder (PNN): s_t -> o_t
        self.obs_decoder = hydra.utils.instantiate(
            obs_decoder_cfg,
            input_dim = state_dim,
            output_dim = obs_dim,
            _recursive_ = False,
        )
        self.rnn_type = rnn_type.lower()
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        # rnn for actionable states: (sa_t, a_t) -> sa_{t+1}
        sa_mem_input_dim = sa_dim + action_dim if self.action_encoder is None else sa_dim + action_ft_dim
        self.sa_memory_unit = rnn_class(sa_mem_input_dim, 
            sa_dim,
            rnn_num_layers,
            batch_first=True).to(self.device)
        # rnn for non-actionable states: (sac_t) -> sac_{t+1}
        self.sac_memory_unit = rnn_class(sac_dim, 
            sac_dim,
            rnn_num_layers,
            batch_first=True).to(self.device)
        self.rnn_num_layers = rnn_num_layers
        # decoder for actionable states (PNN): ha_t -> sa_t
        self.sa_decoder = hydra.utils.instantiate(
            sa_decoder_cfg,
            input_dim = sa_dim,
            output_dim = sa_dim,
            _recursive_ = False
        )
        # decoder for non-actionable states (PNN): hac_t -> sac_t
        self.sac_decoder = hydra.utils.instantiate(
            sac_decoder_cfg,
            input_dim = sac_dim,
            output_dim = sac_dim,
            _recursive_ = False
        )
        # mutual information neural estimator (MINE): (sa_t, a_t, s_{t-1}) -> I(sa_t;a_t|s_{t-1})
        mine_y_dim = action_dim if self.action_encoder is None else action_ft_dim
        self.mine_sa = MINE(
            x_dim = sa_dim + state_dim,
            y_dim = mine_y_dim,
            hidden_dim = mine_hidden_size,
            alpha = mine_alpha
        )
        # mutual information neural estimator (MINE): (sac_t, a_t, s_{t-1}) -> I(sac_t;a_t|s_{t-1})
        self.mine_sac = MINE(
            x_dim = sac_dim + state_dim,
            y_dim = mine_y_dim,
            hidden_dim = mine_hidden_size,
            alpha = mine_alpha
        )

        self.world_modules = [
            self.obs_encoder,
            self.obs_decoder,
            self.sa_memory_unit,
            self.sac_memory_unit,
            self.sa_decoder,
            self.sac_decoder
        ]
        if self.action_encoder is not None:
            self.world_modules.append(self.action_encoder)

        self.mine_modules = [
            self.mine_sa,
            self.mine_sac
        ]
        self.mi_steps = mi_steps
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.obs_weight = obs_weight
        self.kl_weight = kl_weight
        self.mi_weight = mi_weight
        self._metrics = {
            'EV': SequentialExplainedVariance(),
            'IndvEV': SequentialExplainedVariance('raw_values'),
        }
        self._record_history = True
        self._hidden_state_sa = None
        self._hidden_state_sac = None
        self._warm_up_period = 0

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.
        Predict o_{t+1} given (o_t, a_t).

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                * o_t: (Batch_size, Sequence Length, obs_dim)
                * a_t: (Batch_size, Sequence Length, action_dim)
                * delta_o_{t+1}: (Batch_size, Sequence Length, obs_dim)
                * mask: (Batch_size, Sequence Length)

        Returns:
            Dictionary of name to tensor.
        """
        obs, action, _, _ = batch
        if self.action_encoder is not None:
            action = self.action_encoder(action)
        state_mean, state_logvar = self.obs_encoder(obs)

        sa_mean, sac_mean = torch.split(state_mean, [self.sa_dim, self.sac_dim], dim=-1)
        sa_logvar, sac_logvar = torch.split(state_logvar, [self.sa_dim, self.sac_dim], dim=-1)

        # sa_sample = (torch.randn_like(sa_mean) * ((0.5 * sa_logvar).exp()) + sa_mean)
        # sac_sample = (torch.randn_like(sac_mean) * ((0.5 * sac_logvar).exp()) + sac_mean)

        sa_mem_out = self.sa_memory_unit(torch.cat([sa_mean, action], dim=-1))[0]
        sac_mem_out = self.sac_memory_unit(sac_mean)[0]

        sa_next_mean, sa_next_logvar = self.sa_decoder(sa_mem_out)
        sac_next_mean, sac_next_logvar = self.sac_decoder(sac_mem_out)

        # sa_next_sample = (torch.randn_like(sa_next_mean) * ((0.5 * sa_next_logvar).exp()) + sa_next_mean)
        # sac_next_sample = (torch.randn_like(sac_next_mean) * ((0.5 * sac_next_logvar).exp()) + sac_next_mean)

        obs_next = self.obs_decoder(torch.cat([sa_next_mean, sac_next_mean], dim=-1))

        return {
            'prediction': obs_next,
            'sa_prior_next': (sa_next_mean, (0.5*sa_next_logvar).exp()),
            'sac_prior_next': (sac_next_mean, (0.5*sac_next_logvar).exp()),

            'sa_posterior': (sa_mean, (0.5*sa_logvar).exp()),
            'sac_posterior': (sac_mean, (0.5*sac_logvar).exp())
        }
    
    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            net_out: Dictionary containing various outputs from the model.
            batch: The batch passed into the network. This is expected to be a tuple
                * o_t: (Batch_size, Sequence Length, obs_dim)
                * a_t: (Batch_size, Sequence Length, action_dim)
                * delta_o_{t+1}: (Batch_size, Sequence Length, obs_dim)
                * mask: (Batch_size, Sequence Length, 1)
        """
        # prediction loss
        bound_loss = 0.0
        obs_loss, obs_mse_loss = self._obs_loss(net_out['prediction'], batch)

        # kl-divergences
        kl_sa, kl_sac = self._kl_loss(net_out['sa_prior_next'], 
                                      net_out['sac_prior_next'],
                                      net_out['sa_posterior'], 
                                      net_out['sac_posterior'])

        # MI loss
        if self.action_encoder is not None:
            action = self.action_encoder(batch[1])
        else:
            action = batch[1]
        mi_sa, mi_sac = self._mi_loss(net_out['sa_posterior'][0], net_out['sac_posterior'][0], action)

        loss = self.obs_weight*obs_loss + bound_loss \
            + self.obs_weight*obs_mse_loss \
            + self.kl_weight*(kl_sa + kl_sac) \
            + self.mi_weight*(torch.clamp(mi_sac,min=0) - torch.clamp(mi_sa,min=0))

        stats = dict(
            nll = obs_loss.item(),
            mse = obs_mse_loss.item(),
            kl_sa = kl_sa.item(),
            kl_sac = kl_sac.item(),
            mi_sa = mi_sa.item(),
            mi_sac = mi_sac.item(),
            loss = loss.item()
        )

        return loss, stats

    def _obs_loss(self, pred, batch):
        pred_mean = pred[0]
        pred_logvar = pred[1]
        true, mask = batch[2:]
        if self._labels_are_velocities:
            # convert predictions to deltas
            pred_mean = pred_mean - batch[0]

        obs_dist = Normal(pred_mean, (0.5*pred_logvar[1]).exp())
        obs_loss = -torch.mean(obs_dist.log_prob(true))
        obs_mse_loss = F.mse_loss(pred_mean, true)

        return obs_loss, obs_mse_loss

    def _kl_loss(self, prior_sa, prior_sac, posterior_sa, posterior_sac):
        # predicted next step dist.
        prior_sa_dist = Normal(prior_sa[0][:, :-1], prior_sa[1][:, :-1])
        prior_sac_dist = Normal(prior_sac[0][:, :-1], prior_sac[1][:, :-1])

        # actual next step dist.
        posterior_sa_dist = Normal(posterior_sa[0][:, 1:], posterior_sa[1][:, 1:])
        posterior_sac_dist = Normal(posterior_sac[0][:, 1:], posterior_sac[1][:, 1:])

        kl_sa = kl_divergence(prior_sa_dist, posterior_sa_dist)
        kl_sac = kl_divergence(prior_sac_dist, posterior_sac_dist)

        return torch.mean(kl_sa), torch.mean(kl_sac)

    def _mi_loss(self, sa, sac, action):
        a_t = action[:, 1:]
        s_t_1 = torch.cat([sa[:, :-1], sac[:, :-1]], dim=-1).detach()
        mi_sa = self.mine_sa(torch.cat([sa[:, 1:], s_t_1], dim=-1), a_t)
        mi_sac = self.mine_sac(torch.cat([sac[:, 1:], s_t_1], dim=-1), a_t)
        
        return mi_sa, mi_sac

    def configure_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        world_opt = torch.optim.AdamW(
            get_parameters(self.world_modules),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        mi_opt = torch.optim.AdamW(
            get_parameters(self.mine_modules),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return world_opt, mi_opt

    def training_step(self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        world_opt, mi_opt = self.optimizers()

        batch = self.normalizer.normalize_batch(batch)
        net_out = self.get_net_out(batch)

        # train the MI estimator
        for _ in range(self.mi_steps):
            if self.action_encoder is not None:
                action = self.action_encoder(batch[1]).detach()
            else:
                action = batch[1]
            mi_sa, mi_sac = self._mi_loss(net_out['sa_posterior'][0].detach(), 
                                          net_out['sac_posterior'][0].detach(), 
                                          action)
            mi_opt.zero_grad()
            self.manual_backward(mi_sac - mi_sa)
            torch.nn.utils.clip_grad_norm_(get_parameters(self.mine_modules), self.grad_clip_norm)
            mi_opt.step()

        # train the world model
        loss, loss_dict = self.loss(net_out, batch)

        world_opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(get_parameters(self.world_modules), self.grad_clip_norm)
        world_opt.step()

        self._log_stats(loss_dict, prefix='train')
        return loss

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network. This is expected to be a tuple
                * o_t: (Batch_size, obs_dim)
                * a_t: (Batch_size, action_dim)

        Returns:
            The predictions for a single function sample.
        """
        obs, action = (x.unsqueeze(1) for x in net_in)
        batch_size = obs.shape[0]
        if self._hidden_state_sa is None:
            if self.rnn_type == 'gru':
                self._hidden_state_sa = torch.zeros(self.rnn_num_layers, batch_size,
                                                 self.sa_dim, device=self.device)
            else:
                self._hidden_state_sa = tuple(
                    torch.zeros(self.rnn_num_layers, batch_size,
                                self.sa_dim, device=self.device)
                    for _ in range(2))
        if self._hidden_state_sac is None:
            if self.rnn_type == 'gru':
                self._hidden_state_sac = torch.zeros(self.rnn_num_layers, batch_size,
                                                 self.sac_dim, device=self.device)
            else:
                self._hidden_state_sac = tuple(
                    torch.zeros(self.rnn_num_layers, batch_size,
                                self.sac_dim, device=self.device)
                    for _ in range(2))
        with torch.no_grad():
            state_mean, state_logvar = self.obs_encoder(obs)
            if self.action_encoder is not None:
                action = self.action_encoder(action)

            sa_mean, sac_mean = torch.split(state_mean, [self.sa_dim, self.sac_dim], dim=-1)
            sa_logvar, sac_logvar = torch.split(state_logvar, [self.sa_dim, self.sac_dim], dim=-1)

            # sa_sample = (torch.randn_like(sa_mean) * ((0.5 * sa_logvar).exp()) + sa_mean)
            # sac_sample = (torch.randn_like(sac_mean) * ((0.5 * sac_logvar).exp()) + sac_mean)

            sa_mem_out, sa_hidden_out = self.sa_memory_unit(
                    torch.cat([sa_mean, action], dim=-1),
                    self._hidden_state_sa
            )
            sac_mem_out, sac_hidden_out = self.sac_memory_unit(
                sac_mean,
                self._hidden_state_sac
            )
            if self._record_history:
                self._hidden_state_sa = sa_hidden_out
                self._hidden_state_sac = sac_hidden_out

            sa_next_mean, sa_next_logvar = self.sa_decoder(sa_mem_out)
            sac_next_mean, sac_next_logvar = self.sac_decoder(sac_mem_out)

            # sa_next_sample = (torch.randn_like(sa_next_mean) * ((0.5 * sa_next_logvar).exp()) + sa_next_mean)
            # sac_next_sample = (torch.randn_like(sac_next_mean) * ((0.5 * sac_next_logvar).exp()) + sac_next_mean)

            obs_decoder_in = torch.cat([sa_next_mean, sac_next_mean], dim=-1)
            obs_next_mean, obs_next_logvar = \
                (output.squeeze(1) for output in self.obs_decoder(obs_decoder_in))
        
        std_predictions = (0.5 * obs_next_logvar).exp()
        predictions = torch.randn_like(obs_next_mean)*std_predictions + obs_next_mean
        info = {'predictions': predictions,
                'mean_predictions': obs_next_mean,
                'std_predictions': std_predictions}

        return predictions, info

    def multi_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network. This is expected to be a tuple
                * o_t: (Batch_size, obs_dim)
                * a_t: (Batch_size, action_dim)

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(net_in)

    def _get_test_and_validation_metrics(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute additional metrics to be used for validation/test only.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            A dictionary of additional metrics.
        """
        to_return = {}
        pred = net_out['prediction'][0]
        if self._labels_are_velocities:
            pred = pred - batch[0]
        # if 'prediction' in net_out:
        #     pred = net_out['prediction']
        # elif 'mean' in net_out:
        #     pred = net_out['mean']
        # else:
        #     raise ValueError('Need either prediction or mean in the net_out')
        # if len(batch) > 3:  # Check if we are doing RL data or (x, y, mask) data.
        #     raise NotImplementedError('RL data needs to be reimplemented.')
        yi, mask = batch[2:]
        for metric_name, metric in self.metrics.items():
            metric_value = metric(pred, yi, mask)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_{self._dim_name_map[dim_idx]}'] = metric_v
            else:
                to_return[metric_name] = metric_value
        return to_return

    def _normalize_prediction_input(self, model_input: torch.Tensor) -> torch.Tensor:
        """Normalize the input for prediction.

        Args:
            model_input: The input tuple to the model (o_t, a_t)
                o_t : observation at time step t
                a_t : action at time step t

        Returns:
            The normalized input.
        """
        if self.normalize_inputs:
            normed_obs = self.normalizer.normalize(model_input[0], 0)
            normed_action = self.normalizer.normalize(model_input[1], 1)
            return (normed_obs, normed_action)
        return model_input

    def _unnormalize_prediction_output(self, output: torch.Tensor) -> torch.Tensor:
        """Unnormalize the output (next-state) of the model.

        Args:
            output: The output of the model.

        Returns:
            The unnormalized output.
        """
        if self.unnormalize_outputs:
            return self.normalizer.unnormalize(output, 0)
        return output

    def predict(
            self,
            model_input: Tuple[np.ndarray],
            each_input_is_different_sample: Optional[bool] = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions using the currently set sampling method.

        Args:
            model_input: The input to be given to the model.
            each_input_is_different_sample: Whether each input should be treated
                as being drawn from a different sample of the model. Note that this
                may not have an effect on all models (e.g. PNN)

        Returns:
            The output of the model and give a dictionary of related quantities.
        """
        model_input = (torch.Tensor(model_input[0]).to(self.device),
                       torch.Tensor(model_input[1]).to(self.device))
        model_input = self._normalize_prediction_input(model_input)
        if each_input_is_different_sample:
            output, infos = self.multi_sample_output_from_torch(model_input)
        else:
            output, infos = self.single_sample_output_from_torch(model_input)
        output = self._unnormalize_prediction_output(output)
        return output.cpu().numpy(), infos
    
    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        return self._metrics

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def record_history(self) -> bool:
        """Whether to keep track of the quantities being fed into the neural net."""
        return self._record_history

    @record_history.setter
    def record_history(self, mode: bool) -> None:
        """Set whether to keep track of quantities being fed into the neural net."""
        self._record_history = mode

    def clear_history(self) -> None:
        """Clear the history."""
        self._hidden_state_sa = None
        self._hidden_state_sac = None

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def warm_up_period(self) -> int:
        """Amount of data to take in before starting to predict"""
        return self._warm_up_period

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        if self._sample_mode not in [sampling_modes.SAMPLE_FROM_DIST,
                                     sampling_modes.RETURN_MEAN]:
            raise ValueError(
                f'PNN sample mode must either be {sampling_modes.SAMPLE_FROM_DIST} '
                f'or {sampling_modes.RETURN_MEAN}, but received {mode}.')
        self._sample_mode = mode
