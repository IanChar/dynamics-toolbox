import os, sys
from typing import Sequence, Callable, Tuple, Any
from argparse import Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from mlp import MLP
from losses import (
    batch_cali_loss,
    batch_qr_loss,
    batch_interval_loss,
)


""" Utility functions """


def _gather_loss_per_q(loss_fn, model, y, x, q_list, device, args):
    loss_list = []
    for q in q_list:
        q_loss = loss_fn(model, y, x, q, device, args)
        loss_list.append(q_loss)
    loss = torch.mean(torch.stack(loss_list))

    return loss


def _get_loss_function(loss_function):
    if loss_function == "calibration":
        return batch_cali_loss
    elif loss_function == "pinball":
        return batch_qr_loss
    elif loss_function == "interval":
        return batch_interval_loss


class QuantileModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_sizes: Sequence[int],
        hidden_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        loss_function: str = "calibration",
        num_quantile_draws: int = 30,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        args: Namespace = None,
    ):
        super().__init__()
        self.model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=encoder_hidden_sizes,
            hidden_activation=hidden_activation,
        )

        # Get loss function and optimizer fields
        self.loss_function = _get_loss_function(loss_function)
        self.num_quantile_draws = num_quantile_draws
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.args = args

    def get_q_list(self):
        if hasattr(self.args, "fixed_q_list"):
            q_list = self.args.fixed_q_list.to(self.device)
        else:
            q_list = torch.rand(self.num_quantile_draws).to(self.device)

        return q_list

    def forward(
        self,
        x: torch.Tensor,
        q_list: torch.Tensor = None,
        recal_model: Any = None,
        recal_type: str = None,
    ) -> torch.Tensor:
        """
        Get output for given list of quantiles

        :param x: tensor, of size (num_x, dim_x)
        :param q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
        :param recal_model:
        :param recal_type:
        :return:
        """

        if q_list is None:
            q_list = torch.linspace(0.01, 0.99, 99)
        else:
            q_list = q_list.flatten()

        num_x = x.shape[0]
        num_q = q_list.shape[0]

        cdf_preds = []
        for p in q_list:
            if recal_model is not None:
                if recal_type == "torch":
                    recal_model.cpu()  # keep recal model on cpu
                    with torch.no_grad():
                        in_p = recal_model(p.reshape(1, -1)).item()
                elif recal_type == "sklearn":
                    in_p = float(recal_model.predict(p.flatten()))
                else:
                    raise ValueError("recal_type incorrect")
            else:
                in_p = float(p)
            p_tensor = (in_p * torch.ones(num_x)).reshape(-1, 1)
            cdf_in = torch.cat([x, p_tensor], dim=1).to(self.device)
            cdf_pred = self.model(cdf_in)  # shape (num_x, 1)
            cdf_preds.append(cdf_pred)

        pred_mat = torch.cat(cdf_preds, dim=1)  # shape (num_x, num_q)
        assert pred_mat.shape == (num_x, num_q)
        return pred_mat

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:

        x_data, y_data = batch
        q_list = self.get_q_list()
        loss = self.loss_function(self.model, y_data, x_data, q_list, self.args)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x_data, y_data = batch
        q_list = self.get_q_list()
        loss = self.loss_function(self.model, y_data, x_data, q_list, self.args)
        self.log("validation_loss", loss)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        x_data, y_data = batch
        q_list = self.get_q_list()
        loss = self.loss_function(self.model, y_data, x_data, q_list, self.args)
        self.log("test_loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

    def print_device(self):
        device_list = []
        for idx in range(len(self.best_va_model)):
            if next(self.best_va_model[idx].parameters()).is_cuda:
                device_list.append("cuda")
            else:
                device_list.append("cpu")
        print(device_list)

    # def on_train_batch_start(self):


if __name__ == "__main__":
    # sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from data import synthetic_sine_heteroscedastic as data_fn

    def make_synthetic_sine_dataloader(num_data, batch_size, num_workers):
        mean, std, y, x = data_fn(num_data)
        x_tensor = torch.Tensor(x.reshape(-1,1))
        y_tensor = torch.Tensor(y.reshape(-1,1))
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        return dataloader, x_tensor, y_tensor

    train_dataloader, _, _ = make_synthetic_sine_dataloader(1000, 64, 8)
    val_dataloader, _, _ = make_synthetic_sine_dataloader(1000, 1000, 8)
    _, test_x, test_y = make_synthetic_sine_dataloader(1000, 1000, 8)


    args = Namespace(
        scale=True,
        sharp_penalty=None,
    )

    model = QuantileModel(
        input_dim=2,
        output_dim=1,
        encoder_hidden_sizes=[64, 64],
        # loss_function='calibration',
        loss_function='pinball',
        args=args
    )

    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=200,
        check_val_every_n_epoch=10,
        log_every_n_steps=1,
        # callbacks=[EarlyStopping('validation_loss')]
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    from metrics_calibration_quantile import (
        quantile_mean_absolute_calibration_error,
        quantile_sharpness,
    )
    ece = quantile_mean_absolute_calibration_error(
        model=model,
        x=test_x,
        y=test_y,
    )
    sharpness = quantile_sharpness(
        model=model,
        x=test_x,
    )
    print(ece)
    print(sharpness)



