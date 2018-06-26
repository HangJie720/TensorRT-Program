#!/bin/bash

python tensorrt.py --frozen_graph=/container_dir/TensorRT-Program/vgg19/frozen_vgg_19.pb \
  --image_file=image.jpg --int8 --output_dir=/container_dir/TensorRT-Program/vgg19 \
  --input_node=input \
  --output_node=vgg_19/fc8/squeezed

#python tensorrt.py --frozen_graph=/container_dir/TensorRT-Program/inception_v3/frozen_inception_v3.pb \
#  --image_file=image.jpg --native --fp32 --fp16 --int8 --output_dir=/container_dir/TensorRT-Program/inception_v3 \
#  --input_node=input \
#  --output_node=InceptionV3/Predictions/Reshape_1