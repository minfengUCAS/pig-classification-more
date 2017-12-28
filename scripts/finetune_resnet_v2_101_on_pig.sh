#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes a ResNetV1-50 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_typhoon.sh
set -e

# Where the pre-trained ResNetV1-50 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=~/dataset/checkpoints

# Model_name
MODEL_NAME=resnet_v2_101

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=~/dataset/pig-models/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=~/dataset/Pig

# Set which GPU to use
GPU_OPT=1

# model_date
MODEL_DATE=2017_04_14

## Download the pre-trained checkpoint.
#if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
#  mkdir ${PRETRAINED_CHECKPOINT_DIR}
#fi
#if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt ]; then
#  wget http://download.tensorflow.org/models/${MODEL_NAME}_${MODEL_DATE}.tar.gz
#  tar -xvf ${MODEL_NAME}_${MODEL_DATE}.tar.gz
#  mv ${MODEL_NAME}.ckpt ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt
#  rm ${MODEL_NAME}_${MODEL_DATE}.tar.gz
#fi
#
## Download the dataset
#CUDA_VISIBLE_DEVICES=${GPU_OPT} python download_and_convert_data.py \
#  --dataset_name=pig \
#  --dataset_dir=${DATASET_DIR}
#
## Fine-tune only the new layers for 1000 steps.
#CUDA_VISIBLE_DEVICES=${GPU_OPT} python train_image_classifier.py \
#  --train_dir=${TRAIN_DIR} \
#  --dataset_name=pig \
#  --dataset_split_name=train \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=${MODEL_NAME} \
#  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt \
#  --checkpoint_exclude_scopes=resnet_v2_101/logits \
#  --trainable_scopes=resnet_v2_101/logits \
#  --max_number_of_steps=3000 \
#  --batch_size=32 \
#  --learning_rate=0.01 \
#  --save_interval_secs=300 \
#  --save_summaries_secs=300 \
#  --log_every_n_steps=10 \
#  --optimizer=rmsprop \
#  --weight_decay=0.00004
#
## Run evaluation.
#CUDA_VISIBLE_DEVICES=${GPU_OPT} python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=pig \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=${MODEL_NAME}
#
# Run test.
# CUDA_VISIBLE_DEVICES=${GPU_OPT} python test_image_classifier.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${TRAIN_DIR} \
#   --dataset_name=pig \
#   --dataset_split_name=test \
#   --dataset_dir=${DATASET_DIR} \
#   --model_name=${MODEL_NAME} \
#   --output_dir='./pig_finetune_result'

# Fine-tune all the new layers for 3000 steps.  
CUDA_VISIBLE_DEVICES=${GPU_OPT} python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=pig \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=5000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=300 \
  --save_summaries_secs=300 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
CUDA_VISIBLE_DEVICES=${GPU_OPT} python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=pig \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}


# Run test.
CUDA_VISIBLE_DEVICES=${GPU_OPT} python test_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=pig \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}
