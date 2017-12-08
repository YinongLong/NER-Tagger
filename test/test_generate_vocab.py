# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
sys.path.insert(0, '/data/longyinong/NER-Tagger')

from utils import generate_vocabulary

data_path = '/data/longyinong/NER-Tagger/data/ner.train.data'
tags_save = '../data/tags_vocab.txt'
text_save = '../data/text_vocab.txt'

generate_vocabulary(data_path, tags_save, text_save)
