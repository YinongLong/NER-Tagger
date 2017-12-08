# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
sys.path.insert(0, '/data/longyinong/NER-Tagger')

from utils import extract_ner_data

raw_data_path = '/data1/public/yubo/multitask_DNer/newdata/DNer/valid.data'
save_path = '../data/ner.valid.data'

# with codecs.open(raw_data_path, 'rb', 'utf8') as data_f:
# 	count = 0
# 	for line in data_f:
# 		line = line.rstrip()
# 		# line = line.replace('\r\n', 'A')
# 		print(line.split(), '-', len(line))
# 		count += 1
# 		if count >= 10:
# 			break
# 	pass

extract_ner_data(raw_data_path, save_path)