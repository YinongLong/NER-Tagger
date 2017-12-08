#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

running_type=1

train_data=./data/ner.train.data
validation_data=./data/ner.valid.data
text_vocab_path=./data/text_vocab.txt
tags_vocab_path=./data/tags_vocab.txt

text_vocab_size=0
tags_vocab_size=0
embedding_dim=128
hidden_dim=256
projection_dim=100
text_rnn_layers=1
context_rnn_layers=1
text_rnn_dropout=0.
context_rnn_dropout=0.

epochs=30
batch_size=128
lr=0.001
logging_interval=20
validation_interval=100

model_save=./ner_model/ed${embedding_dim}_hd${hidden_dim}_pd${projection_dim}_trl${text_rnn_layers}\
_crl${context_rnn_layers}_trd${text_rnn_dropout}_crd${context_rnn_dropout}_bs${batch_size}_lr${lr}
mkdir -p ${model_save}


if [ ${running_type} -le 0 ]; then
    echo training model ...
    python train_ner_model.py \
      --train-data ${train_data} \
      --validation-data ${validation_data} \
      --text-vocab-path ${text_vocab_path} \
      --tags-vocab-path ${tags_vocab_path} \
      --model-save ${model_save} \
      --cuda-available \
      --text-vocab-size ${text_vocab_size} \
      --tags-vocab-size ${tags_vocab_size} \
      --embedding-dim ${embedding_dim} \
      --hidden-dim ${hidden_dim} \
      --projection-dim ${projection_dim} \
      --bidirectional \
      --text-rnn-layers ${text_rnn_layers} \
      --context-rnn-layers ${context_rnn_layers} \
      --text-rnn-dropout ${text_rnn_dropout} \
      --context-rnn-dropout ${context_rnn_dropout} \
      --epochs ${epochs} \
      --batch-size ${batch_size} \
      --lr ${lr} \
      --logging-interval ${logging_interval} \
      --validation-interval ${validation_interval}
fi

test_data=./data/ner.test.data
save_result_path=./data/test_result.txt

echo testing ...
python generate.py \
  --test-data ${test_data} \
  --text-vocab-path ${text_vocab_path} \
  --tags-vocab-path ${tags_vocab_path} \
  --model-save ${model_save} \
  --save-result-path ${save_result_path} \
  --cuda-available \
  --embedding-dim ${embedding_dim} \
  --hidden-dim ${hidden_dim} \
  --projection-dim ${projection_dim} \
  --bidirectional \
  --text-rnn-layers ${text_rnn_layers} \
  --context-rnn-layers ${context_rnn_layers} \
  --batch-size ${batch_size}