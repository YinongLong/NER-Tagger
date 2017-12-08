#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

train_data=./data/tags.data
valid_data=./data/tags.valid.data
ner_train_data=./data/ner.train.data
tags_vocab_path=./data/tags_vocab.txt
text_vocab_path=./data/text_vocab.txt

## training setting
batch_size=512
epochs=30
early_stop=5
lr=0.001
clipping_norm=5.0
logging_interval=20
validation_interval=100
dropout=0.3

## model setting
vocab_size=0
embedding_dim=60
hidden_dim=128
projection_dim=100
n_layers=1

model_save_path=./vs${vocab_size}_ed${embedding_dim}_hd${hidden_dim}_pd${projection_dim}\
_nl${n_layers}_dp${dropout}_lr${lr}_bs${batch_size}_model
mkdir -p ${model_save_path}

python train_tag_lm.py \
  --train-data ${train_data} \
  --valid-data ${valid_data} \
  --ner-train-data ${ner_train_data} \
  --tags-vocab-path ${tags_vocab_path} \
  --text-vocab-path ${text_vocab_path} \
  --vocab-size ${vocab_size} \
  --batch-size ${batch_size} \
  --epochs ${epochs} \
  --early-stop ${early_stop} \
  --lr ${lr} \
  --clipping-norm ${clipping_norm} \
  --logging-interval ${logging_interval} \
  --validation-interval ${validation_interval} \
  --model-save-path ${model_save_path} \
  --embedding-dim ${embedding_dim} \
  --hidden-dim ${hidden_dim} \
  --projection-dim ${projection_dim} \
  --n-layers ${n_layers} \
  --dropout ${dropout}
