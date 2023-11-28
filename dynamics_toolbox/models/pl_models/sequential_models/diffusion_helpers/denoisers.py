"""
Diffusion Model architecture helper functions and classes

Author: Aravind Venugopal
"""
import torch
import torch.nn as nn
import numpy as np

from dynamics_toolbox.models.pl_models.sequential_models.diffusion_helpers.utils import TimeSiren, TransformerEncoderBlock, FCBlock

class Model_mlp_diff_embed(nn.Module):
    # this model embeds x, y, t, before input into a fc NN (w residuals)
    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        embed_dim,
        output_dim=None,
        is_dropout=False,
        is_batch=False,
        activation="relu",
        net_type="fc",
        use_prev=False,
    ):
        super(Model_mlp_diff_embed, self).__init__()
        self.embed_dim = embed_dim  # input embedding dimension
        self.n_hidden = n_hidden
        self.net_type = net_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_prev = use_prev  # whether x contains previous timestep
        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # embedding NNs
        if self.use_prev:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(int(x_dim / 2), self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(x_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )  # no prev hist
        self.y_embed_nn = nn.Sequential(
            nn.Linear(y_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.t_embed_nn = TimeSiren(1, self.embed_dim)

        # fc nn layers
        if self.net_type == "fc":
            if self.use_prev:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 4, n_hidden))  # concat x, x_prev,
            else:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 3, n_hidden))  # no prev hist
            self.fc2 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))  # will concat y and t at each layer
            self.fc3 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))
            self.fc4 = nn.Sequential(nn.Linear(n_hidden + y_dim + 1, self.output_dim))

        # transformer layers
        elif self.net_type == "transformer":
            self.nheads = 16  # 16
            self.trans_emb_dim = 64
            self.transformer_dim = self.trans_emb_dim * self.nheads  # embedding dim for each of q,k and v (though only k and v have to be same I think)

            self.t_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.y_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.x_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)

            self.pos_embed = TimeSiren(1, self.trans_emb_dim)

            self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block2 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block3 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block4 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)

            if self.use_prev:
                self.final = nn.Linear(self.trans_emb_dim * 4, self.output_dim)  # final layer params
            else:
                self.final = nn.Linear(self.trans_emb_dim * 3, self.output_dim)
        else:
            raise NotImplementedError

    def forward(self, y, x, t, context_mask):
        # embed y, x, t

       
        if self.use_prev:
            x_e = self.x_embed_nn(x[:, :int(self.x_dim / 2)])
            x_e_prev = self.x_embed_nn(x[:, int(self.x_dim / 2):])
        else:
            x_e = self.x_embed_nn(x)  # no prev hist
            x_e_prev = None
        y_e = self.y_embed_nn(y)
        t_e = self.t_embed_nn(t)

        # mask out context embedding, x_e, if context_mask == 1
        context_mask = context_mask.repeat(x_e.shape[-1], 1, 1).T #WILL NEED TO FIX THIS

        #x_e = x_e * (-1 * (1 - context_mask))
        if self.use_prev:
            x_e_prev = x_e_prev * (-1 * (1 - context_mask))

        # pass through fc nn
        if self.net_type == "fc":
            net_output = self.forward_fcnn(x_e, x_e_prev, y_e, t_e, x, y, t)

        # or pass through transformer encoder
        elif self.net_type == "transformer":
            net_output = self.forward_transformer(x_e, x_e_prev, y_e, t_e, x, y, t)

        return net_output

    def forward_fcnn(self, x_e, x_e_prev, y_e, t_e, x, y, t):

        batch_size, seq_len = x.shape[0], x.shape[1]

        x_e = x_e.view(batch_size * seq_len, -1)
        if x_e_prev is not None:
          x_e_prev = x_e_prev.view(batch_size * seq_len, -1)
        y_e = y_e.view(batch_size * seq_len, -1)
        t_e = t_e.view(batch_size * seq_len, -1)
        x = x.view(batch_size * seq_len, -1)
        y = y.view(batch_size * seq_len, -1)
        t = t.view(batch_size * seq_len, -1)
        if self.use_prev:
            net_input = torch.cat((x_e, x_e_prev, y_e, t_e), 1)
        else:
            net_input = torch.cat((x_e, y_e, t_e), 1)
        nn1 = self.fc1(net_input)
        nn2 = self.fc2(torch.cat((nn1 / 1.414, y, t), 1)) + nn1 / 1.414  # residual and concat inputs again
        nn3 = self.fc3(torch.cat((nn2 / 1.414, y, t), 1)) + nn2 / 1.414
        net_output = self.fc4(torch.cat((nn3, y, t), 1))
        net_output = net_output.view(batch_size, seq_len, -1)
        return net_output

    def forward_transformer(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        # roughly following this: https://jalammar.github.io/illustrated-transformer/

        ######################### RESHAPING #########################3
        batch_size, seq_len = x.shape[0], x.shape[1]

        x_e = x_e.view(batch_size * seq_len, -1)
        if x_e_prev is not None:
          x_e_prev = x_e_prev.view(batch_size * seq_len, -1)
        y_e = y_e.view(batch_size * seq_len, -1)
        t_e = t_e.view(batch_size * seq_len, -1)
        x = x.view(batch_size * seq_len, -1)
        y = y.view(batch_size * seq_len, -1)
        t = t.view(batch_size * seq_len, -1)

        #############################################################3
        t_input = self.t_to_input(t_e)
        y_input = self.y_to_input(y_e)
        x_input = self.x_to_input(x_e)
        if self.use_prev:
            x_input_prev = self.x_to_input(x_e_prev)
        # shape out = [batchsize, trans_emb_dim]

        # add 'positional' encoding
        # note, here position refers to order tokens are fed into transformer
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)
        if self.use_prev:
            x_input_prev += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 4.0)

        if self.use_prev:
            inputs1 = torch.cat(
                (
                    t_input[None, :, :],
                    y_input[None, :, :],
                    x_input[None, :, :],
                    x_input_prev[None, :, :],
                ),
                0,
            )
        else:
            inputs1 = torch.cat((t_input[None, :, :], y_input[None, :, :], x_input[None, :, :]), 0)
        # shape out = [3, batchsize, trans_emb_dim]

        #print("inputs1 shape", inputs1.shape)

        block1 = self.transformer_block1(inputs1)
        block2 = self.transformer_block2(block1)
        block3 = self.transformer_block3(block2)
        block4 = self.transformer_block4(block3)

        # flatten and add final linear layer
        # transformer_out = block2
        transformer_out = block4
        transformer_out = transformer_out.transpose(0, 1)  # roll batch to first dim
        # shape out = [batchsize, 3, trans_emb_dim]
        #print("transformer out shape", transformer_out.shape)
        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        # shape out = [batchsize, 3 x trans_emb_dim]

        out = self.final(flat)
        # shape out = [batchsize, n_dim]
        #print("out shape", out.shape)
        out = out.view(batch_size, seq_len, -1)
        #print("out shape", out.shape)
        return out