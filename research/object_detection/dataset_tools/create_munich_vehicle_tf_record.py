# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Convert raw Munich vehicle dataset to TFRecord for object_detection.

Example usage:

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import PIL.Image
import tensorflow as tf
import cv2
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', ' ', 'Root directory to raw Munich Vehicle dataset.')
flags.DEFINE_string('set', '', 'Convert training set, validation set or '
                               'merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'test']
EXAMPLES_PATHS = {'train': 'Train_crop',
                  'test': 'Test_crop'}

#
"""
vehicle_types = {'bus': 30, 'cam': 20, 'pkw_trail': 11, 'pkw': 10, 'truck': 22, 'truck_trail': 23, 'van_trail': 17}

% Used class: 
% Car: ca + van = pkw => 10,16 => make both as 10 
% Truck: truck + cam => 20,22 = > make both as 20

Ignore the rest classes. 
"""


# Experiment 1: Ground truth labels from website starts with mapping: 10-> car, 20 -> truck
# OBJ_NAME = {10: 'car',
#             20: 'truck'}
# # USED LABELS starts from 1.
# USE_LABEL = {10: 1,
#              20: 2}

# Experiment 2: Assign all vehicles as one type:
OBJ_NAME = {30: 'vehicle', 20: 'vehicle', 11: 'vehicle', 10: 'vehicle', 22: 'vehicle', 23: 'vehicle', 17: 'vehicle'}
# USED LABELS starts from 1.
USE_LABEL = {30: 1, 20: 1, 11: 1, 10: 1, 22: 1, 23: 1, 17: 1}


def dict_to_tf_example(split_data_dir,
                       name,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert Munich datset to tf.Example proto.
    Convert the image "name" of the Munich dataset (train/val) into record file
    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    IMG_TYPE = '.jpg'
    GT_EXT = '_gt.txt'
    file_name = name + IMG_TYPE
    img_path = os.path.join(split_data_dir, file_name)
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    # Get the image shape information
    tmp_img = cv2.imread(img_path)
    height, width, depth = tmp_img.shape

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    # Get the ground truth bounding box file
    gt_filename = os.path.join(split_data_dir, name + GT_EXT)
    with open(gt_filename) as f:
        lines = f.read().splitlines()
    for line in lines:
        difficult_obj.append(int(0))
        line_float = [float(item) for item in line.split(',')]
        label = int(line_float[-1])
        xmin.append(line_float[0] / width)  # xmin
        ymin.append(line_float[1] / height)  # ymin
        xmax.append(line_float[2] / width)  # xmax
        ymax.append(line_float[3] / height)  # ymax
        classes_text.append(OBJ_NAME[label].encode('utf8'))
        classes.append(USE_LABEL[label])
        # Add zeros for truncated (not being used for Munich dataset)
        truncated.append(0)
        poses.append('none'.encode('utf8'))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            file_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            file_name.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    # List of example images for the set (train/val)
    examples_path = os.path.join(data_dir, EXAMPLES_PATHS[FLAGS.set], FLAGS.set + '.txt')
    examples_list = dataset_util.read_examples_list(examples_path)
    print(len(examples_list))
    for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples_list))
        split_data_dir = os.path.join(data_dir, EXAMPLES_PATHS[FLAGS.set])
        tf_example = dict_to_tf_example(split_data_dir, name=example,
                                        ignore_difficult_instances=FLAGS.ignore_difficult_instances)
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
