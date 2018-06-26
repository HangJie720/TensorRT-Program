#!/bin/bash

python export_inference_graph.py \
  --alsologtostderr \
  --model_name=vgg_19 \
  --output_file=/container_dir/TensorRT-Program/vgg19/vgg_19_graph.pb \
  --labels_offset=1
