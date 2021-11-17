"""
Utility for loss functions.
"""
from typing import Callable, Union
from argparse import Namespace

import torch

import dynamics_toolbox.constants.losses as losses


def get_regression_loss(
        name: str,
        **kwargs
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get a regression loss function based on name.

    Args:
        name: The name of the loss.
        kwargs: Any other named arguments to pass to the loss function.

    Returns:
        The loss function.
    """
    if name == losses.MSE:
        return torch.nn.MSELoss(**kwargs)
    elif name == losses.MAE:
        return torch.nn.L1Loss(**kwargs)
    else:
        raise ValueError(f'Unknown loss {name}.')


def get_quantile_loss(
        name: str,
        **kwargs
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get a quantile loss function based on name.

    Args:
        name: The name of the loss.
        kwargs: Any other named arguments to pass to the loss function.

    Returns:
        The loss function.
    """
    if name == losses.CALI:
        return get_cali_loss(**kwargs)
    elif name == losses.PB:
        return get_pinball_loss(**kwargs)
    elif name == losses.INT:
        return get_interval_loss(**kwargs)
    else:
        raise ValueError(f'Unknown loss {name}.')


""" Quantile Loss Definitions """

def get_cali_loss(
        should_scale: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the calibration loss. As introduced in https://arxiv.org/abs/2011.09588

    Args:
        should_scale: Whether the loss should be scaled magnitude of difference
            between true coverage and intended coverage.

    Returns:
        Callable calibration loss.
    """
    def cali_loss(
            y: torch.Tensor,
            q_pred: torch.Tensor,
            q_list: torch.Tensor,
    ) -> torch.Tensor:
        """Calibration loss function for Quantile Models.

        Args:
            y: torch tensor of the targets, shape (N,1) (must be 1 dimensional)
            q_pred: torch tensor of quantile predictions, shape (N, Q)
            q_list: torch tensor of quantile levels, shape (Q,)

        Returns:
            The calibration loss value.
        """
        num_pts = y.size(0)
        num_q = q_list.size(0)
        assert num_q.shape == (num_q,)
        y_mat = y.repeat(num_q, 1).T
        assert y_mat.shape == (num_pts, num_q)
        idx_under = (y_mat <= q_pred).reshape(num_pts, num_q)
        idx_over = ~idx_under
        coverage = torch.mean(idx_under.float(), dim=0)  # shape (num_q,)
        diff_mat = y_mat - q_pred  # shape  (num_pts, num_q)
        mean_diff_under = torch.mean(-1 * diff_mat * idx_under, dim=1)
        mean_diff_over = torch.mean(diff_mat * idx_over, dim=1)
        cov_under = coverage < q_list
        cov_over = ~cov_under
        loss_list = (cov_under * mean_diff_over) + (cov_over * mean_diff_under)
        # handle scaling
        if should_scale:
            cov_diff = torch.abs(coverage - q_list)
            loss_list = cov_diff * loss_list
            loss = torch.mean(loss_list)
        else:
            loss = torch.mean(loss_list)
        #TODO: Possibly add in sharpness penalty.
        return loss

    return cali_loss


def get_pinball_loss(**kwargs) \
        -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the pinball loss.

    Returns:
        The pinball loss.
    """
    def pinball_loss(
            y: torch.Tensor,
            q_pred: torch.Tensor,
            q_list: torch.Tensor,
    ) -> torch.Tensor:
        """Pinball loss function for Quantile Models.

        Args:
            y: torch tensor of the targets, shape (N,1) (must be 1 dimensional)
            q_pred: torch tensor of quantile predictions, shape (N, Q)
            q_list: torch tensor of quantile levels, shape (Q,)

        Returns:
            The pinball loss value.
        """
        num_pts = y.size(0)
        num_q = q_list.size(0)
        assert q_list.shape == (num_q,)
        q_rep = q_list.view(-1, 1).repeat(1, num_pts).T
        y_mat = y.repeat(1, num_q)
        assert y_mat.shape == (num_pts, num_q)
        diff = q_pred - y_mat
        mask = (diff.ge(0).float() - q_rep).detach()
        loss = (mask * diff).mean()
        return loss
    return pinball_loss


def get_interval_loss(**kwargs) \
        -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get the interval loss.

    Returns:
        The interval loss.
    """
    def interval_loss(
            y: torch.Tensor,
            q_pred: torch.Tensor,
            q_list: torch.Tensor,
    ) -> torch.Tensor:
        """Interval loss (negative interval score) function for Quantile Models.

        Args:
            y: torch tensor of the targets, shape (N,1) (must be 1 dimensional)
            q_pred: torch tensor of quantile predictions, shape (N, Q)
            q_list: torch tensor of quantile levels, shape (Q,)

        Returns:
            The interval loss value.
        """
        num_pts = y.size(0)
        num_q = q_list.size(0)
        # make sure that the quantile levels are complementary
        #   i.e. q_list: [0.03, 0.4, 0.12, ..., 0.97, 0.6, 0.88, ...]
        assert num_q % 2 == 0
        l_list = q_list[:num_q//2]
        u_list = q_list[num_q//2:]
        assert torch.max((1-l_list) - u_list) < 1e-10
        num_l = num_q//2
        l_pred = q_pred[:, num_l]  # should be shape (N, Q//2)
        u_pred = q_pred[:, num_l:]  # should be shape (N, Q//2)
        below_l = (l_pred - y.view(-1)).gt(0)
        above_u = (y.view(-1) - u_pred).gt(0)
        loss = (
            (u_pred - l_pred)
            + (1.0 / l_list).view(-1, 1)
            * (l_pred - y.view(-1))
            * below_l
            + (1.0 / l_list).view(-1, 1)
            * (y.view(-1) - u_pred)
            * above_u
        )
        return torch.mean(loss)
    return interval_loss

