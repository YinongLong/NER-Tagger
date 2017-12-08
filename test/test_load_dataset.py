# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
sys.path.insert(0, '/data/longyinong/NER-Tagger')

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

dataset = load_dataset(ner_data_path, text_vocab.word2idx, tags_vocab.word2idx)
print(dataset[3])

dataset = load_dataset(tag_data_path, tags_vocab=tags_vocab.word2idx)
print(dataset[0])
