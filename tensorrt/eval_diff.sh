#!/bin/bash

python test.py --frozen_graph=/container_dir/TensorRT-Program/vgg19/frozen_vgg_19.pb \
  --native --fp32 --fp16 --int8 --output_dir=/container_dir/TensorRT-Program/vgg19 \
  --input_node=input \
  --output_node=vgg_19/fc8/squeezed \
  --batch_size=32 \
  --ids_are_one_indexed
