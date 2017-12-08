# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class TagSequenceModel(nn.Module):

	def __init__(self, tags_num, embedding_dim, hidden_dim, projection_dim, num_layers=1, dropout=0.):
		"""
		:param tags_num: int
			tag vocabulary size
		:param embedding_dim: int
			tag embedding dimension
		:param hidden_dim: int
			dimension of hidden state of RNNs
		:param projection_dim: int
			dimension of projection layer
		:param num_layers: int
		:param dropout: float
			probability of tag embedding drop
		"""
		super(TagSequenceModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.tags_num = tags_num

		self.embedding = nn.Embedding(tags_num, embedding_dim)
		self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers)
		self.projection_layer = nn.Linear(hidden_dim, projection_dim)
		self.output_layer = nn.Linear(projection_dim, tags_num)
		self.dropout_wrapper = nn.Dropout(dropout)

	def forward(self, inputs, length, hidden_state):
		"""
		:param inputs: Variable, [max_seq_len x batch_size]
		:param length: array, [batch_size]
		:param hidden_state: Variable, [num_layers x batch_size x hidden_dim]
		:return:
			projection_out: Variable, [max_seq_len x batch_size x tags_num]
			hidden_state: Variable, [num_layers x batch_size x hidden_dim]
		"""
		inputs = self.embedding(inputs)  # [max_seq_len x batch_size x embedding_dim]
		inputs = self.dropout_wrapper(inputs)

		sort_indexes = np.argsort(-length)
		unsort_indexes = np.argsort(sort_indexes)

		inputs = inputs[:, sort_indexes, :]
		length = length[sort_indexes]
		if hidden_state is not None:  # Note using sort in hidden state
			hidden_state = hidden_state[:, sort_indexes, :]

		inputs = pack_padded_sequence(inputs, length)

		# output [max_seq_len x batch_size x hidden_dim]
		# hidden_state [num_layers x batch_size x hidden_dim]
		output, hidden_state = self.gru(inputs, hidden_state)
		output, _ = pad_packed_sequence(output)
		output = output[:, unsort_indexes, :]
		hidden_state = hidden_state[:, unsort_indexes, :]

		output_list = []
		batch_size = output.size(1)
		for i in range(batch_size):
			single_sample = output[:, i, :]
			single_projection = self.projection_layer(single_sample)
			single_output = self.output_layer(single_projection)
			output_list.append(single_output.unsqueeze(1))
		output = torch.cat(output_list, 1)
		return output, hidden_state

	def init_hidden_state(self, batch_size):
		weight = next(self.parameters()).data
		hidden_state = Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
		return hidden_state
