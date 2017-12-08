# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import torch
import codecs

from torch.autograd import Variable
from ner_model import NERModel
from utils import Vocabulary
from utils import load_model
from utils import load_dataset
from utils import get_data_worker


parser = argparse.ArgumentParser('NER Recognizer')

parser.add_argument('--test-data', default='', type=str, required=True,
                    help='path of training data')
parser.add_argument('--text-vocab-path', default='', type=str, required=True,
                    help='path of saving text vocabulary')
parser.add_argument('--tags-vocab-path', default='', type=str, required=True,
                    help='path of saving tags vocabulary')
parser.add_argument('--model-save', default='', type=str, required=True,
                    help='directory of saving trained model')
parser.add_argument('--save-result-path', default='', type=str, required=True,
                    help='path of saving prediction results')

parser.add_argument('--cuda-available', action='store_true',
                    help='if CUDA environment is available')
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

parser.add_argument('--batch-size', default=128, type=int,
                    help='batch size')

args = parser.parse_args()

# loading vocabulary
text_vocab = Vocabulary(args.text_vocab_path)
tags_vocab = Vocabulary(args.tags_vocab_path)
text_vocab.load()
tags_vocab.load()


model = NERModel(text_vocab.word_num, tags_vocab.word_num, args.embedding_dim, args.hidden_dim,
                 args.projection_dim, args.bidirectional, args.text_rnn_layers, args.context_rnn_layers, 0., 0.)
load_model(model, os.path.join(args.model_save, 'model.data'))

if args.cuda_available:
	model.cuda()
model.eval()

test_data = load_dataset(args.test_data, text_vocab.word2idx, tags_vocab.word2idx)
container, worker = get_data_worker(test_data, False, args.batch_size, 20, True)
worker.start()


def write_prediction(file_obj, inputs_data, predict_data, target_data):
	"""
	:param file_obj: file object
	:param inputs_data: LongTensor, [max_seq_len x batch_size]
	:param predict_data: LongTensor, [max_seq_len x batch_size]
	:param target_data: LongTensor, [max_seq_len x batch_size]
	:return: None
	"""
	# check data type and shape
	assert isinstance(inputs_data, torch.LongTensor)
	assert isinstance(predict_data, torch.LongTensor)
	assert isinstance(target_data, torch.LongTensor)
	assert inputs_data.size() == predict_data.size()
	assert predict_data.size() == target_data.size()
	max_seq_len = inputs_data.size(0)
	batch_size = inputs_data.size(1)
	for i in range(batch_size):
		for j in range(max_seq_len):
			try:
				if inputs_data[j, i] == 1:  # PAD token ID
					break
				tokens = list()
				tokens.append(text_vocab.idx2word[inputs_data[j, i]])
				tokens.append(tags_vocab.idx2word[target_data[j, i]])
				tokens.append(tags_vocab.idx2word[predict_data[j, i]])
				out_line = ' '.join(tokens) + '\n'
				file_obj.write(out_line)
			except Exception as e:
				print(inputs_data.size(), predict_data.size(), target_data.size())
				raise e
		# delimiter of instances
		file_obj.write('\n')


save_result_file = codecs.open(args.save_result_path, 'wb')

for batch in range(1, worker.num_batches + 1):
	inputs, targets, lengths = container.get(timeout=10)
	if args.cuda_available:
		inputs = inputs.cuda()
	inputs = Variable(inputs, volatile=True)
	prediction = model(inputs, lengths)
	_, indexes = torch.topk(prediction, 1, 1)
	indexes = indexes.cpu().view(*targets.size())
	targets = targets.contiguous()
	inputs = inputs.cpu().contiguous()
	write_prediction(save_result_file, inputs.data, indexes.data, targets)

save_result_file.close()
print('testing process finish!!!')

