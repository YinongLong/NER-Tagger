# -*- coding: utf-8 -*-
from __future__ import print_function

import Queue
import sys
sys.path.insert(0, '/data/longyinong/NER-Tagger')

from utils import get_data_worker
from utils import load_dataset
from utils import Vocabulary

text_vocab_save_path = '/data/longyinong/NER-Tagger/data/text_vocab.txt'
tags_vocab_save_path = '/data/longyinong/NER-Tagger/data/tags_vocab.txt'

text_vocab = Vocabulary(text_vocab_save_path)
text_vocab.load()

tags_vocab = Vocabulary(tags_vocab_save_path)
tags_vocab.load()

ner_data_path = '/data/longyinong/NER-Tagger/data/ner.train.data'
tag_data_path = '/data/longyinong/NER-Tagger/data/tags.data'

ner_dataset = load_dataset(ner_data_path, text_vocab.word2idx, tags_vocab.word2idx)

container, worker = get_data_worker(ner_dataset, True, 128, 20)
print('num of batches: %d' % worker.num_batches)
worker.start()

consumed = 0
try:
	while True:
		source, target, length = container.get(timeout=10)
		print(source.size(), target.size(), length.shape)
		consumed += 1
except Queue.Empty:
	print('the number of all batches is %d' % worker.num_batches)
	print('%d batches have been put into container ...' % worker.num_produced)
	print('we have consumed %d batches ...' % consumed)
