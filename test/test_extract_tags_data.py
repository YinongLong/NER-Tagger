# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
sys.path.insert(0, '/data/longyinong/NER-Tagger')

from utils import extract_tags_data

raw_data_path = '/data1/public/yubo/multitask_DNer/newdata/DNer/valid.data'
save_path = '../data/tags.valid.data'

extract_tags_data(raw_data_path, save_path)
