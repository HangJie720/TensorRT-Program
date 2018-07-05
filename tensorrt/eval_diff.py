# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Methods for running the Official Models with TensorRT.

Please note that all of these methods are in development, and subject to
rapid change.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import imghdr
import json
import os
import sys
import time
import math
import data_provider
import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader
import tensorflow.contrib.tensorrt as trt

import imagenet_preprocessing

_GPU_MEM_FRACTION = 0.25
_WARMUP_NUM_LOOPS = 5
_LOG_FILE = "log.txt"
_LABELS_FILE = "labellist.json"
_GRAPH_FILE = "frozen_graph.pb"


################################################################################
# Prep the image input to the graph.
################################################################################
def preprocess_image(file_name, output_height=224, output_width=224,
                     num_channels=3):
  """Run standard ImageNet preprocessing on the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    output_height: int, final height of image
    output_width: int, final width of image
    num_channels: int, depth of input image

  Returns:
    Float array representing processed image with shape
      [output_height, output_width, num_channels]

  Raises:
    ValueError: if image is not a JPEG.
  """
  if imghdr.what(file_name) != "jpeg":
    raise ValueError("At this time, only JPEG images are supported. "
                     "Please try another image.")

  image_buffer = tf.read_file(file_name)
  normalized = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=None,
      output_height=output_height,
      output_width=output_width,
      num_channels=num_channels,
      is_training=False)

  with tf.Session(config=get_gpu_config()) as sess:
    result = sess.run([normalized])

  return result[0]


def write_graph_to_file(graph_name, graph_def, output_dir):
  """Write Frozen Graph file to disk."""
  output_path = os.path.join(output_dir, graph_name)
  with tf.gfile.GFile(output_path, "wb") as f:
    f.write(graph_def.SerializeToString())


################################################################################
# Utils for handling Frozen Graphs.
################################################################################
def get_serving_meta_graph_def(savedmodel_dir):
  """Extract the SERVING MetaGraphDef from a SavedModel directory.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.

  Returns:
    MetaGraphDef that should be used for tag_constants.SERVING mode.

  Raises:
    ValueError: if a MetaGraphDef matching tag_constants.SERVING is not found.
  """
  # We only care about the serving graph def
  tag_set = set([tf.saved_model.tag_constants.SERVING])
  serving_graph_def = None
  saved_model = reader.read_saved_model(savedmodel_dir)
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == tag_set:
      serving_graph_def = meta_graph_def
  if not serving_graph_def:
    raise ValueError("No MetaGraphDef found for tag_constants.SERVING. "
                     "Please make sure the SavedModel includes a SERVING def.")

  return serving_graph_def


def convert_savedmodel_to_frozen_graph(savedmodel_dir, output_dir):
  """Convert a SavedModel to a Frozen Graph.

  A SavedModel includes a `variables` directory with variable values,
  and a specification of the MetaGraph in a ProtoBuffer file. A Frozen Graph
  takes the variable values and inserts them into the graph, such that the
  SavedModel is all bundled into a single file. TensorRT and TFLite both
  leverage Frozen Graphs. Here, we provide a simple utility for converting
  a SavedModel into a frozen graph for use with these other tools.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.
    output_dir: string representing path to the output directory for saving
      the frozen graph.

  Returns:
    Frozen Graph definition for use.
  """
  meta_graph = get_serving_meta_graph_def(savedmodel_dir)
  signature_def = tf.contrib.saved_model.get_signature_def_by_key(
      meta_graph,
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

  outputs = [v.name for v in signature_def.outputs.itervalues()]
  output_names = [node.split(":")[0] for node in outputs]

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(
        sess, meta_graph.meta_info_def.tags, savedmodel_dir)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_names)

  write_graph_to_file(_GRAPH_FILE, frozen_graph_def, output_dir)

  return frozen_graph_def


def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


################################################################################
# Run the graph in various precision modes.
################################################################################
def get_gpu_config():
  """Share GPU memory between image preprocessing and inference."""
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
  return tf.ConfigProto(gpu_options=gpu_options)


def equal(x, y):
    count=0.0
    for i in range(len(x)):
        if x[i] == y[i]:
            count += 1
    return count
def equal_1(x,y):
    count=0.0
    for i in range(len(x)):
        for id in x[i]:
            if id == y[i]:
                count += 1
                break
    return count

def execute_graph(mode, graph_def, image_list, label_list, input_node, output_node, flags):
  # Run the inference graph.
  tf.logging.info("Starting execution")
  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=graph_def,
        return_elements=[input_node, output_node]
    )
    # Unwrap the returned output node. For now, we assume we only
    # want the tensor with index `:0`, which is the 0th element of the
    # `.outputs` list.
    inp = inp.outputs[0]
    out = out.outputs[0]
  img_list = []
  lab_list = []
  image_arrays = []
  accuracy = 0.0
  # for image_path, label in zip(image_list[5000:6000], label_list[5000:6000]):
  #     image_array = preprocess_image(image_path, 224, 224, 3)
  #     image_arrays.append(image_array)
  # np.save("imagenet_val_10000.npy",image_arrays)

  # image_5000 = np.load("imagenet_val_5000.npy")
  # image_10000 = np.load("imagenet_val_10000.npy")
  # image_arrays.append(image_5000)
  # image_arrays.append(image_10000)
  # image_arrays = np.reshape(image_arrays,[10000,224,224,3])
  # image_arrays = np.array(image_arrays)
  # np.save("imagenet_val_5000_10000.npy",image_arrays)
  # print(image_arrays.shape)

  image_arrays = np.load("imagenet_val_5000_10000.npy")
  with tf.Session(graph=g, config=get_gpu_config()) as sess:
      nb_batches = int(math.ceil(float(len(image_arrays)) / flags.batch_size))
      assert nb_batches * flags.batch_size >= len(image_arrays)
      for batch in range(nb_batches):
          start = batch * flags.batch_size
          end = min(len(image_arrays), start + flags.batch_size)
          cur_batch_size = end - start

          img_list[:cur_batch_size] = image_arrays[start:end]
          lab_list[:cur_batch_size] = label_list[start:end]
          feed_dict = {inp: img_list}

          cur_corr_preds = sess.run(out, feed_dict=feed_dict)
          accuracy += equal_1(top_predictions(cur_corr_preds[:cur_batch_size],1, flags.ids_are_one_indexed),lab_list)
      assert end >= len(image_arrays)
      accuracy /= len(image_arrays)

  print('{} Top1 Accuracy: {:.5f}'.format(mode, accuracy))


################################################################################
# Parse predictions
################################################################################
def get_labels():
  """Get the set of possible labels for classification."""
  with open(_LABELS_FILE, "r") as labels_file:
    labels = json.load(labels_file)

  return labels


def top_predictions(result, n, ids_are_one_indexed=False):
  """Get the top n predictions given the array of softmax results."""
  # We only care about the first example.
  ids_array = []
  for i in range(len(result)):
      probabilities = result[i]
      # Get the ids of most probable labels. Reverse order to get greatest first.
      ids = np.argsort(probabilities)[::-1]
      ids_array.append(ids[:n]+int(ids_are_one_indexed))
  return ids_array


################################################################################
# Run this script
################################################################################
def main(argv):
  parser = TensorRTParser()
  flags = parser.parse_args(args=argv[1:])

  # Load the data.
  image_list, label_list = data_provider.prepare_sample_list(
      'imagenet/val/','imagenet/val_new.txt')

  # Load the graph def
  if flags.frozen_graph:
    frozen_graph_def = get_frozen_graph(flags.frozen_graph)
  elif flags.savedmodel_dir:
    frozen_graph_def = convert_savedmodel_to_frozen_graph(
        flags.savedmodel_dir, flags.output_dir)
  else:
    raise ValueError(
        "Either a Frozen Graph file or a SavedModel must be provided.")


  # Run inference in all desired modes.
  if flags.native:
    mode = "native"
    print("Running {} graph".format(mode))
    execute_graph(mode,
                  frozen_graph_def,
                  image_list,
                  label_list,
                  flags.input_node,
                  flags.output_node,
                  flags)

  if flags.fp32:
    mode = "FP32"
    print("Running {} graph".format(mode))
    # graph = run_trt_graph_for_mode(graph_name, frozen_graph_def, mode, flags)
    fp32_graph = get_frozen_graph(flags.optimized_graph)
    execute_graph(mode,
                  fp32_graph,
                  image_list,
                  label_list,
                  flags.input_node,
                  flags.output_node,
                  flags)

  if flags.fp16:
    mode = "FP16"
    print("Running {} graph".format(mode))
    # graph = run_trt_graph_for_mode(graph_name, frozen_graph_def, mode, flags)
    fp16_graph = get_frozen_graph(flags.optimized_graph)
    execute_graph(mode,
                  fp16_graph,
                  image_list,
                  label_list,
                  flags.input_node,
                  flags.output_node,
                  flags)

  if flags.int8:
    mode = "INT8"
    print("Running {} graph".format(mode))
    int8_graph = get_frozen_graph(flags.optimized_graph)
    execute_graph(mode,
                  int8_graph,
                  image_list,
                  label_list,
                  flags.input_node,
                  flags.output_node,
                  flags)



class TensorRTParser(argparse.ArgumentParser):
  """Parser to contain flags for running the TensorRT timers."""

  def __init__(self):
    super(TensorRTParser, self).__init__()

    self.add_argument(
        "--frozen_graph", "-fg", default=None,
        help="[default: %(default)s] The location of a Frozen Graph "
        "protobuf file that will be used for inference. Note that either "
        "savedmodel_dir or frozen_graph should be passed in, and "
        "frozen_graph will take precedence.",
        metavar="<FG>",
    )
    self.add_argument(
        "--optimized_graph", "-og", default=None,
        help="[default: %(default)s] The location of a optimized Graph "
             "protobuf file that will be used for inference. Note that either "
             "savedmodel_dir or frozen_graph should be passed in, and "
             "frozen_graph will take precedence.",
        metavar="<FG>",
    )

    self.add_argument(
        "--savedmodel_dir", "-sd", default=None,
        help="[default: %(default)s] The location of a SavedModel directory "
        "to be converted into a Frozen Graph. This is equivalent to the "
        "subdirectory that is created under the directory specified by "
        "--export_dir when running an Official Model. Note that either "
        "savedmodel_dir or frozen_graph should be passed in, and "
        "frozen_graph will take precedence.",
        metavar="<SD>",
    )

    self.add_argument(
        "--output_node", "-on", default="softmax_tensor",
        help="[default: %(default)s] The names of the graph output node "
        "that should be used when retrieving results. Assumed to be a softmax.",
        metavar="<ON>",
    )

    self.add_argument(
        "--input_node", "-in", default="input_tensor",
        help="[default: %(default)s] The name of the graph input node where "
        "the float image array should be fed for prediction.",
        metavar="<ON>",
    )

    self.add_argument(
        "--batch_size", "-bs", type=int, default=128,
        help="[default: %(default)s] Batch size for inference. If an "
        "image file is passed, it will be copied batch_size times to "
        "imitate a batch.",
        metavar="<BS>"
    )

    self.add_argument(
        "--native", action="store_true",
        help="[default: %(default)s] If set, benchmark the model "
        "with it's native precision and without TensorRT."
    )

    self.add_argument(
        "--fp32", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using fp32 precision."
    )

    self.add_argument(
        "--fp16", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using fp16 precision."
    )

    self.add_argument(
        "--int8", action="store_true",
        help="[default: %(default)s] If set, benchmark the model with TensorRT "
        "using int8 precision."
    )

    self.add_argument(
        "--workspace_size", "-ws", type=int, default=2<<10,
        help="[default: %(default)s] Workspace size in megabytes.",
        metavar="<WS>"
    )

    self.add_argument(
        "--ids_are_one_indexed", action="store_true",
        help="[default: %(default)s] Some ResNet models include a `background` "
        "category, and others do not. If the model used includes `background` "
        "at index 0 in the output and represents all 1001 categories, "
        "this should be False. If the model used omits the `background` label "
        "and has only 1000 categories, this should be True."
    )

    self.add_argument(
        "--predictions_to_print", "-pp", type=int, default=5,
        help="[default: %(default)s] Number of predicted labels to predict.",
        metavar="<PP>"
    )


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
