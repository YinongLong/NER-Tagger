# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import time

import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from tag_lm import TagSequenceModel
from utils import generate_vocabulary
from utils import Vocabulary
from utils import load_dataset
from utils import get_data_worker
from utils import save_model
from utils import load_model
from utils import EarlyStoppingError
from utils import tally_parameters


parser = argparse.ArgumentParser('Tag Language Modeling')
parser.add_argument('--train-data', default='', type=str,
                    help='path of training data')
parser.add_argument('--valid-data', default='', type=str,
                    help='path of validation data')
parser.add_argument('--ner-train-data', default='', type=str,
                    help='path of ner training data')
parser.add_argument('--tags-vocab-path', default='', type=str,
                    help='path of tags vocabulary')
parser.add_argument('--text-vocab-path', default='', type=str,
                    help='path of text vocabulary')
parser.add_argument('--vocab-size', default=0, type=int,
                    help='default value (0 representing all unique tags)')
parser.add_argument('--batch-size', default=128, type=int,
                    help='batch size')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of training epochs')
parser.add_argument('--early-stop', default=5, type=int,
                    help='early stopping')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--clipping-norm', default=5.0, type=float,
                    help='clipping gradients norm for prevent gradients exploding')
parser.add_argument('--cuda-available', action='store_true', default=True,
                    help='if cuda environment is available')
parser.add_argument('--logging-interval', default=20, type=int,
                    help='interval of logging training process')
parser.add_argument('--validation-interval', default=100, type=int,
                    help='interval of validation')
parser.add_argument('--resume', action='store_true',
                    help='resume training at trained model')
parser.add_argument('--model-save-path', default='', type=str,
                    help='path of saving model')

parser.add_argument('--embedding-dim', default=60, type=int,
                    help='dimension of tag embedding')
parser.add_argument('--hidden-dim', default=128, type=int,
                    help='dimension of hidden unit in RNNs')
parser.add_argument('--projection-dim', default=100, type=int,
                    help='dimension of projection layer')
parser.add_argument('--n-layers', default=1, type=int,
                    help='number of layers in RNNs')
parser.add_argument('--dropout', default=0., type=float,
                    help='probability of dropout')


args = parser.parse_args()


def batch_train(model, inputs, targets, length, criterion):
	"""
	:param model: nn.Module
	:param inputs: Variable, [max_seq_len x batch_size]
	:param targets: Variable, [max_seq_len x batch_size]
	:param length: array
	:param criterion: nn.CrossEntropyLoss
	:return:
	"""
	hidden_state = model.init_hidden_state(inputs.size(1))
	vocab_logit, _ = model(inputs, length, hidden_state)
	loss = criterion(vocab_logit.view(-1, model.tags_num), targets.view(-1))
	return loss


def train(m_args, model, training_data, validation_data):
	"""
	:param m_args: arguments of model
	:param model: nn.Module
	:param training_data: list
	:param validation_data: list
	:return: None
	"""
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=m_args.lr)
	criterion = nn.CrossEntropyLoss(ignore_index=1)

	minimum_val_loss = None
	minimum_unchanged = 0

	try:
		for i in range(1, m_args.epochs + 1):

			container, worker = get_data_worker(training_data, True, m_args.batch_size, 20)
			worker.start()

			total_loss_val = 0.
			start_time = time.time()
			samples_count = 0
			total_samples = worker.total_num_samples

			for batch in range(1, worker.num_batches + 1):
				source, target, length = container.get(timeout=10)
				samples_count += source.size(1)
				if m_args.cuda_available:
					source = source.cuda()
					target = target.cuda()
				source = Variable(source)
				target = Variable(target)
				optimizer.zero_grad()

				loss = batch_train(model, source, target, length, criterion)
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
							i, samples_count, total_samples, elapsed_time * 1000 / samples_count,
							current_loss_val, np.exp(current_loss_val)
						))

				if batch % m_args.validation_interval == 0:
					validation_start_time = time.time()
					validation_loss = validation(model, validation_data, m_args.batch_size, criterion,
					                             m_args.cuda_available)
					if minimum_val_loss is None or minimum_val_loss > validation_loss:
						minimum_val_loss = validation_loss
						minimum_unchanged = 0
						save_model(model, os.path.join(m_args.model_save_path, 'model.data'))
					else:
						minimum_unchanged += 1

					print('-' * 75)
					print('| trained after {:10d} dialogues | validation loss {:5.2f} | ppl {:5.2f}\n'
					      '| minimum validation loss {:5.2f} | minimum ppl {:5.2f}'.format(
						samples_count + (i - 1) * total_samples,
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

	container, worker = get_data_worker(validation_data, False, batch_size, 20)
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
	assert os.path.exists(args.ner_train_data)
	assert os.path.exists(args.train_data)

	# checking existing of tags vocabulary
	if not os.path.exists(args.tags_vocab_path):
		generate_vocabulary(args.ner_train_data, args.tags_vocab_path, args.text_vocab_path)

	# loading tags vocabulary
	vocab = Vocabulary(args.tags_vocab_path)
	vocab.load()

	# loading training data set and validation data set
	tags_train_dataset = load_dataset(args.train_data, tags_vocab=vocab.word2idx)
	tags_valid_dataset = load_dataset(args.valid_data, tags_vocab=vocab.word2idx)

	# building tags language modeling
	tag_model = TagSequenceModel(vocab.word_num, args.embedding_dim, args.hidden_dim, args.projection_dim)
	print('the number of parameters: %d' % tally_parameters(tag_model))
	if args.cuda_available:
		tag_model.cuda()

	# resume training process
	if args.resume and os.path.exists(os.path.join(args.model_save_path, 'model.data')):
		load_model(tag_model, os.path.join(args.model_save_path, 'model.data'))

	# training model
	train(args, tag_model, tags_train_dataset, tags_valid_dataset)
