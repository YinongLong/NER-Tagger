# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from ner_model import NERModel
from utils import generate_vocabulary
from utils import get_data_worker
from utils import load_dataset
from utils import tally_parameters
from utils import Vocabulary
from utils import EarlyStoppingError
from utils import save_model


parser = argparse.ArgumentParser('NER Recognizer')

parser.add_argument('--train-data', default='', type=str,
                    help='path of training data')
parser.add_argument('--validation-data', default='', type=str,
                    help='path of validation data')
parser.add_argument('--text-vocab-path', default='', type=str,
                    help='path of saving text vocabulary')
parser.add_argument('--tags-vocab-path', default='', type=str,
                    help='path of saving tags vocabulary')
parser.add_argument('--model-save', default='', type=str,
                    help='directory of saving trained model')

parser.add_argument('--cuda-available', action='store_true',
                    help='if CUDA environment is available')
parser.add_argument('--text-vocab-size', default=0, type=int,
                    help='default value (0 representing all unique terms)')
parser.add_argument('--tags-vocab-size', default=0, type=int,
                    help='default value (0 representing all unique tags)')
parser.add_argument('--embedding-dim', default=100, type=int,
                    help='dimension of term embedding')
parser.add_argument('--hidden-dim', default=256, type=int,
                    help='dimension of hidden units')
parser.add_argument('--projection-dim', default=100, type=int,
                    help='dimension of projection layer')
parser.add_argument('--bidirectional', action='store_true',
                    help='if text RNNs is bidirectional')
parser.add_argument('--text-rnn-layers', default=1, type=int,
                    help='number of layers of text RNNs')
parser.add_argument('--context-rnn-layers', default=1, type=int,
                    help='number of layers of context RNNs')
parser.add_argument('--text-rnn-dropout', default=0.3, type=float,
                    help='dropout probability of text RNNs')
parser.add_argument('--context-rnn-dropout', default=0.3, type=float,
                    help='dropout probability of context RNNs')

parser.add_argument('--epochs', default=30, type=int,
                    help='number of epochs')
parser.add_argument('--batch-size', default=128, type=int,
                    help='batch size')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--clipping-norm', default=5.0, type=float,
                    help='maximum norm of gradient')
parser.add_argument('--early-stop', default=5, type=int,
                    help='early stopping')
parser.add_argument('--max-batch-terms', default=30, type=int,
                    help='control the number of terms processed in each batch')
parser.add_argument('--logging-interval', default=20, type=int,
                    help='interval of logging training process')
parser.add_argument('--validation-interval', default=100, type=int,
                    help='interval of validation')

args = parser.parse_args()


def batch_train(model, batch_inputs, batch_targets, lengths, criterion):
	"""
	:param model: nn.Module
	:param batch_inputs: Variable, [max_seq_len x batch_size]
	:param batch_targets: Variable, [max_seq_len x batch_size]
	:param lengths: array
	:param criterion: nn.CrossEntropyLoss
	:return:
	"""
	prediction_logit = model(batch_inputs, lengths)
	loss = criterion(prediction_logit, batch_targets.view(-1))
	return loss


def train(m_args, model, training_data, validation_data):
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=m_args.lr)
	criterion = nn.CrossEntropyLoss(ignore_index=1)

	minimum_val_loss = None
	minimum_unchanged = 0
	try:
		for epoch in range(1, m_args.epochs + 1):
			container, worker = get_data_worker(training_data, True, m_args.batch_size, 20, True)
			worker.start()

			total_loss_val = 0.
			start_time = time.time()
			samples_count = 0
			total_samples = worker.total_num_samples

			for batch in range(1, worker.num_batches + 1):
				batch_inputs, batch_targets, lengths = container.get(timeout=10)
				samples_count += batch_inputs.size(1)
				if m_args.cuda_available:
					batch_inputs = batch_inputs.cuda()
					batch_targets = batch_targets.cuda()
				batch_inputs = Variable(batch_inputs)
				batch_targets = Variable(batch_targets)
				optimizer.zero_grad()

				loss = batch_train(model, batch_inputs, batch_targets, lengths, criterion)
				loss.backward()
				torch.nn.utils.clip_grad_norm(model.parameters(), m_args.clipping_norm)
				optimizer.step()

				total_loss_val += loss.data[0]

				if batch % m_args.logging_interval == 0:
					current_loss_val = total_loss_val / m_args.logging_interval
					total_loss_val = 0.
					elapsed_time = time.time() - start_time
					print(
						'| Epoch {:3d} | {:8d}/{:8d} dialogues | ms/dialog {:5.2f} | loss {:5.2f} | ppl {:5.2f}'.format(
							epoch, samples_count, total_samples, elapsed_time * 1000 / samples_count,
							current_loss_val, np.exp(current_loss_val)
						))

				if batch % m_args.validation_interval == 0:
					validation_start_time = time.time()
					validation_loss = validation(model, validation_data, m_args.batch_size, criterion,
					                             m_args.cuda_available)
					if minimum_val_loss is None or minimum_val_loss > validation_loss:
						minimum_val_loss = validation_loss
						minimum_unchanged = 0
						save_model(model, os.path.join(m_args.model_save, 'model.data'))
					else:
						minimum_unchanged += 1

					print('-' * 75)
					print('| trained after {:10d} dialogues | validation loss {:5.2f} | ppl {:5.2f}\n'
					      '| minimum validation loss {:5.2f} | minimum ppl {:5.2f}'.format(
						samples_count + (epoch - 1) * total_samples,
						validation_loss,
						np.exp(validation_loss),
						minimum_val_loss,
						np.exp(minimum_val_loss)))
					print('-' * 75)
					model.train()
					if minimum_unchanged >= m_args.early_stop:  # stop training
						print('^_^ ' * 19)
						print('Early-Stopping'.center(75, ' '))
						print('^_^ ' * 19)
						raise EarlyStoppingError('early-stopping')
					start_time += time.time() - validation_start_time
	except EarlyStoppingError:
		pass


def validation(model, validation_data, batch_size, criterion, cuda_available):
	model.eval()

	container, worker = get_data_worker(validation_data, False, batch_size, 20, True)
	worker.start()

	total_loss_val = 0.

	for _ in range(worker.num_batches):
		source, target, length = container.get(timeout=10)
		if cuda_available:
			source = source.cuda()
			target = target.cuda()
		source = Variable(source, volatile=True)
		target = Variable(target, volatile=True)

		loss = batch_train(model, source, target, length, criterion)
		total_loss_val += loss.data[0]
	return total_loss_val / worker.num_batches


if __name__ == '__main__':
	if (not os.path.exists(args.text_vocab_path)) or (not os.path.exists(args.tags_vocab_path)):
		generate_vocabulary(args.train_data, args.tags_vocab_path, args.text_vocab_path)
	text_vocab = Vocabulary(args.text_vocab_path)
	tags_vocab = Vocabulary(args.tags_vocab_path)
	text_vocab.load()
	tags_vocab.load()

	train_data = load_dataset(args.train_data, text_vocab.word2idx, tags_vocab.word2idx)
	valid_data = load_dataset(args.validation_data, text_vocab.word2idx, tags_vocab.word2idx)

	ner_model = NERModel(text_vocab.word_num, tags_vocab.word_num, args.embedding_dim, args.hidden_dim,
	                     args.projection_dim, args.bidirectional, args.text_rnn_layers, args.context_rnn_layers,
	                     args.text_rnn_dropout, args.context_rnn_dropout)

	print('the number of parameters: %d' % tally_parameters(ner_model))

	if args.cuda_available:
		ner_model.cuda()

	train(args, ner_model, train_data, valid_data)
