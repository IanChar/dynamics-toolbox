'''
Implementation for Mutual Information Neural Estimator (MINE).
(https://arxiv.org/abs/1801.04062)

Author: Namrata Deka
Date:
'''
import math

import torch
import torch.nn as nn

EPS = 1e-6
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = torch.logsumexp(input, 0).squeeze() - math.log(input.shape[0])
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None

def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema

def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0).squeeze() - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128, alpha=0.01,loss='mine_biased'):
        # loss = ['mine', 'mine_biased']
        super(MINE, self).__init__()
        input_dim = x_dim + y_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.act = nn.ELU()
        self.loss = loss
        self.alpha = alpha
        self.running_mean = 0

    def forward(self, x, y):
        # (x;y|z)
        joint = torch.cat([x, y], dim=-1)
        marginal = torch.cat([x, y[torch.randperm(y.shape[0])]], dim=-1)

        t_joint = self.network(joint).reshape(-1, 1)
        t_marginal = self.network(marginal).reshape(-1, 1)
        # print(t_joint.shape)
        # mi = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))
        if self.loss == 'mine_biased':
            mi = torch.mean(t_joint) - (torch.logsumexp(t_marginal, 0).squeeze() - math.log(t_marginal.shape[0]))
        elif self.loss == 'mine':
            second_term, self.running_mean = ema_loss(
                t_marginal, self.running_mean, self.alpha)
            mi = torch.mean(t_joint) - second_term
        else:
            raise NotImplementedError
        return mi

    def network(self, input):
        h1 = self.act(self.fc1(input))
        h2 = self.act(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3
