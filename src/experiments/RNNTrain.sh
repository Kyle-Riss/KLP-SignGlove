#!/usr/bin/env bash

# ASL-style runner for KLP SignGlove

time_steps=87
batch_size=64
epochs=100
lr=0.001
hidden_size=64

model="MSCSGRU" # Options: MSCSGRU, MSCGRU, CNNGRU, GRU, StackedGRU, LSTM, StackedLSTM, TransformerEncoder, CNNEncoder, HybridEncoder
layers=2
number_heads=4

description="baseline"
project_name="KLP-SignGlove"

python LightningTrain.py \
  -layers $layers \
  -model $model \
  -hidden_size $hidden_size \
  -lr $lr \
  -time_steps $time_steps \
  -batch_size $batch_size \
  -epochs $epochs \
  -number_heads $number_heads \
  -project_name $project_name \
  -description $description



