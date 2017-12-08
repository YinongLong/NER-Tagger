# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import codecs
import os
import math
import threading
import random
import Queue
import collections

import torch

import numpy as np


SOS_token = '<SOS>'
PAD_token = '<PAD>'
UNK_token = '<UNK>'


class Vocabulary(object):

	def __init__(self, save_path):
		self.save_path = save_path
		self.word2freq = collections.Counter()
		self.word2idx = {SOS_token: 0, PAD_token: 1, UNK_token: 2}
		self.idx2word = [SOS_token, PAD_token, UNK_token]
		self.word_num = 3

	def update(self, items):
		for item in items:
			self.word2freq[item] += 1

	def convert(self):
		freq_tuple = self.word2freq.most_common()
		for word, _ in freq_tuple:
			self.word2idx[word] = self.word_num
			self.idx2word.append(word)
			self.word_num += 1

	def save(self):
		word_idx = sorted(self.word2idx.items(), key=lambda item: item[1])
		with open(self.save_path, 'wb') as save_f:
			for word, _ in word_idx:
				word += '\n'
				save_f.write(word)
		print('generated %d words ...' % self.word_num)

	def load(self):
		assert os.path.exists(self.save_path)
		self.word2freq = None
		self.word2idx = {}
		self.idx2word = []
		self.word_num = 0
		with codecs.open(self.save_path, 'rb') as data_f:
			idx = 0
			for line in data_f:
				word = line.strip()
				self.word2idx[word] = idx
				self.idx2word.append(word)
				self.word_num += 1
				idx += 1


def generate_vocabulary(data_path, tags_save_path, text_save_path):
	assert os.path.exists(data_path)

	tags_vocab = Vocabulary(tags_save_path)
	text_vocab = Vocabulary(text_save_path)

	with codecs.open(data_path, 'rb') as data_f:
		for line in data_f:
			text_part, tags_part = line.strip().split('\t')
			text_vocab.update(text_part.strip().split())
			tags_vocab.update(tags_part.strip().split())

	tags_vocab.convert()
	text_vocab.convert()
	tags_vocab.save()
	text_vocab.save()


def extract_tags_data(data_path, save_path):
	assert os.path.exists(data_path)

	print('start extracting tags ...')
	count = 0
	with codecs.open(data_path, 'rb') as data_f,\
		codecs.open(save_path, 'wb') as save_f:

		line_content = []
		for line in data_f:
			tokens = line.strip().split()
			if tokens:
				line_content.append(tokens[-1])
			else:
				if line_content:
					line_content = ' '.join(line_content) + '\n'
					save_f.write(line_content)
					count += 1
				line_content = []
		if line_content:
			line_content = ' '.join(line_content) + '\n'
			save_f.write(line_content)
			count += 1
	print('number of samples extracted from raw data: %d' % count)


def extract_ner_data(data_path, save_path):
	assert os.path.exists(data_path)

	print('start extracting ner data ...')
	with codecs.open(data_path, 'rb') as data_f,\
		codecs.open(save_path, 'wb') as save_f:

		line_text = []
		line_tags = []
		count = 0
		for line in data_f:
			tokens = line.strip().split()
			if tokens:
				if len(tokens) == 2:  # encounter ^M char
					continue
				# check the availability of chars
				text_token = tokens[0].strip()
				tag_token = tokens[-1].strip()
				if text_token and tag_token:
					line_text.append(text_token)
					line_tags.append(tag_token)
			else:  # blank line
				if line_text and line_tags:
					assert len(line_text) == len(line_tags)
					line_content = ' '.join(line_text) + '\t' + ' '.join(line_tags) + '\n'
					save_f.write(line_content)
					count += 1
				line_text = []
				line_tags = []

		if line_text and line_tags:
			assert len(line_text) == len(line_tags)
			line_content = ' '.join(line_text) + '\t' + ' '.join(line_tags) + '\n'
			save_f.write(line_content)
			count += 1
	print('number of samples extracted from raw data: %d' % count)


def load_dataset(data_path, text_vocab=None, tags_vocab=None):
	"""
	:param data_path: str
	:param text_vocab: dict
	:param tags_vocab: dict
	:return:
		dataset: list
	"""
	dataset = []
	with codecs.open(data_path, 'rb') as data_f:
		for line in data_f:
			parts = line.strip().split('\t')
			if len(parts) == 2:  # extract NER data
				try:
					text_part = [text_vocab.get(term, 2) for term in parts[0].strip().split()]
				except Exception as e:
					for term in parts[0].strip().split():
						if term not in text_vocab:
							print(term)
					print(parts[0])
					print(parts[1])
					raise e
				tags_part = [tags_vocab.get(tag, 2) for tag in parts[1].strip().split()]
				text_part = np.array(text_part, np.int64)
				tags_part = np.array(tags_part, np.int64)
				dataset.append((text_part, tags_part))
			elif len(parts) == 1:  # extract Tags data
				tags_part = [tags_vocab[tag] for tag in parts[0].strip().split()]
				tags_part = np.array(tags_part, np.int64)
				dataset.append(tags_part)
			else:
				raise Exception('encountered unexpected data format!')
	return dataset


def get_data_worker(dataset, shuffle, batch_size, max_capacity, decreasing=False):
	container = Queue.Queue(max_capacity)
	worker = DataWorker(container, dataset, shuffle, batch_size, decreasing)
	return container, worker


class DataWorker(threading.Thread):

	def __init__(self, container, dataset, shuffle, batch_size, decreasing):
		"""
		:param container: Queue
		:param dataset: list
		:param shuffle: bool
		:param batch_size: int
		:param decreasing: bool
		"""
		threading.Thread.__init__(self)
		assert isinstance(container, Queue.Queue)
		self.daemon = True
		self.container = container
		self.dataset = dataset
		self.batch_size = batch_size
		self.decreasing = decreasing
		self.num_batches = int(math.ceil(len(dataset) / batch_size))
		self.num_produced = 0
		self.total_num_samples = len(dataset)

		if shuffle:
			random.shuffle(dataset)

	def run(self):
		batch_container = []
		for item in self.dataset:
			batch_container.append(item)
			if len(batch_container) == self.batch_size:
				self.container.put(pack_batch_samples(batch_container, self.decreasing))
				self.num_produced += 1
				batch_container = []
		if batch_container:
			self.container.put(pack_batch_samples(batch_container, self.decreasing))
			self.num_produced += 1


def pack_batch_samples(batch_samples, decreasing):
	batch_size = len(batch_samples)

	length = []
	if isinstance(batch_samples[0], np.ndarray):
		if decreasing:
			batch_samples.sort(key=lambda item: len(item), reverse=True)

		for sample in batch_samples:
			length.append(len(sample))
		max_seq_len = max(length)

		source_ndarray = np.ones((batch_size, max_seq_len), np.int64)
		source_ndarray[:, 0] = 0
		target_ndarray = np.ones((batch_size, max_seq_len), np.int64)
		for idx, sample in enumerate(batch_samples):
			source_ndarray[idx, 1:length[idx]] = sample[:length[idx]-1]
			target_ndarray[idx, :length[idx]] = sample

		length = np.array(length, np.int64)
		source_samples = torch.from_numpy(source_ndarray).long().transpose(0, 1)
		target_samples = torch.from_numpy(target_ndarray).long().transpose(0, 1)
	else:
		if decreasing:
			batch_samples.sort(key=lambda item: len(item[0]), reverse=True)

		for sample in batch_samples:
			length.append(len(sample[0]))
		max_seq_len = max(length)

		source_ndarray = np.ones((batch_size, max_seq_len), np.int64)
		target_ndarray = np.ones((batch_size, max_seq_len), np.int64)
		for idx, (text_part, tags_part) in enumerate(batch_samples):
			source_ndarray[idx, :length[idx]] = text_part
			target_ndarray[idx, :length[idx]] = tags_part

		length = np.array(length, np.int64)

		source_samples = torch.from_numpy(source_ndarray).long().transpose(0, 1)
		target_samples = torch.from_numpy(target_ndarray).long().transpose(0, 1)

	return source_samples, target_samples, length


def save_model(model, save_path):
	torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
	model.load_state_dict(torch.load(save_path))


def tally_parameters(model):
	return sum([p.nelement() for p in model.parameters()])


class EarlyStoppingError(Exception):

	def __init__(self, value):
		self.value = value

	def __str_(self):
		return str(self.value)
