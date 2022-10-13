import os

import torch
import torch.nn.functional as F
from torch.nn import Module, LSTMCell, Embedding, Linear, Dropout, BatchNorm1d


class Attention(Module):

	def __init__(self, input_size, hidden_size, dropout_prob = 0.25) -> None:
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.embedding = Embedding(input_size, hidden_size)
		self.dropout = Dropout(dropout_prob)


	def forward(self, context, hidden):



class RNNEncoder(Module):

	def __init__(self, input_size, hidden_size):

		super().__init__()
		self.hidden_size = hidden_size

		self.embedding = Embedding(input_size, hidden_size)

		self.lstm = LSTM(hidden_size, hidden_size)

	def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoder(Module):


	def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, dropout_prob = 0.25) -> None:
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout_prob = dropout_prob


		self.embedding = Embedding(input_size, hidden_size)
		self.attention = Linear(hidden_size*2, seq_len)
		self.combine_att = Linear(hidden_size*2, hidden_size)
		self.dropout = Dropout(dropout_prob)

		self.LSTM = LSTM(hidden_size, hidden_size, num_layers = num_layers, device=self.device)
		self.out = Linear(hidden_size, output_size)


	def forward(self, input, hidden, encoded):

		embed = self.embedding(input).view(1,1,-1)
		embed = self.dropout(embed)

		attention_weights = F.softmax(
			self.attention(torch.cat((embed[0], hidden[0]),1)), dim=1)

		attention = torch.bmm(attention_weigths.unsqueeze(0), encoded.unsqueeze(0))

		output = torch.cat((embed[0], attention[0]),1)
		output = self.combine_att(output).unsqueeze(0)

		output = F.relu(output)
		output, hidden = self.LSTM(output, hidden)

		output = self.out(output)

		return output, hidden, attn_weights

		output = F.relu(output)
		output, hidden = self.LSTM(output, hidden)
		output = self.out(output[0])

		return output, hidden

	def initHidden(self):
		return torch.zeros(1, 1, self.hidden_size, device=device)






