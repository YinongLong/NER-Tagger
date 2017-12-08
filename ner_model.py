# -*- coding: utf-8 -*-
from __future__ import print_function

import torch.nn as nn


class NERModel(nn.Module):

	def __init__(self, text_vocab_size, tags_vocab_size, embedding_dim, hidden_dim,
	             projection_dim, bidirectional, text_rnn_layers, context_rnn_layers,
	             text_rnn_dropout, context_rnn_dropout):
		super(NERModel, self).__init__()
		self.hidden_dim = hidden_dim

		self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim)
		self.text_rnn_dropout = nn.Dropout(text_rnn_dropout)
		self.text_rnn = nn.GRU(embedding_dim, hidden_dim, text_rnn_layers,
		                       dropout=text_rnn_dropout,
		                       bidirectional=bidirectional)
		self.context_rnn_dropout = nn.Dropout(context_rnn_dropout)
		self.context_rnn = nn.GRU(hidden_dim * (2 if bidirectional else 1), hidden_dim, context_rnn_layers,
		                          dropout=context_rnn_dropout)
		self.projection_layer = nn.Linear(hidden_dim, projection_dim)
		self.prediction_layer = nn.Linear(projection_dim, tags_vocab_size)

	def forward(self, inputs, lengths):
		"""
		:param inputs: Variable, [max_seq_len x batch_size]
		:param lengths: array, [batch_size] (Note: a decreasing order)
		:return:
			prediction: Variable, [max_seq_len * batch_size x tags_vocab_size]
		"""
		inputs = self.text_embedding(inputs)  # [max_seq_len x batch_size x embedding_dim]
		inputs = self.text_rnn_dropout(inputs)
		# text level RNN
		inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths)
		inputs, _ = self.text_rnn(inputs)
		inputs, lengths = nn.utils.rnn.pad_packed_sequence(inputs)  # [max_seq_len x batch_size x hidden_dim * num_directions]
		# context level RNN
		inputs = self.context_rnn_dropout(inputs)
		inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths)
		inputs, _ = self.context_rnn(inputs)
		inputs, _ = nn.utils.rnn.pad_packed_sequence(inputs)  # [max_seq_len x batch_size x hidden_dim]
		# prediction level
		inputs = inputs.view(-1, self.hidden_dim)  # [max_seq_len * batch_size x hidden_dim]
		inputs = self.projection_layer(inputs)  # [max_seq_len * batch_size x projection_dim]
		inputs = self.prediction_layer(inputs)  # [max_seq_len * batch_size x tags_vocab_size]
		return inputs
