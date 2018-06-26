#!/bin/bash

CHECKPOINT_DIR=/container_dir/TensorRT-Program/vgg19/vgg_19.ckpt
DATASET_DIR=/ILSVRC2012

python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=vgg_19 \
    --labels_offset=1
