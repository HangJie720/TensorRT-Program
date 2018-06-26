#!/bin/bash

#python freeze_graph.py \
#  --input_graph=/container_dir/TensorRT-Program/vgg19/vgg_19_graph.pb \
#  --input_checkpoint=/container_dir/TensorRT-Program/vgg19/vgg_19.ckpt \
#  --input_binary=true --output_graph=/container_dir/TensorRT-Program/vgg19/frozen_vgg_19.pb \
#  --output_node_names=vgg_19/fc8/squeezed

python freeze_graph.py \
  --input_graph=/container_dir/TensorRT-Program/inception_v3/inception_v3_inf_graph.pb \
  --input_checkpoint=/container_dir/TensorRT-Program/inception_v3/inception_v3.ckpt \
  --input_binary=true --output_graph=/container_dir/TensorRT-Program/inception_v3/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1