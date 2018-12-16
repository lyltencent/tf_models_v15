# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Converts Munich vechile detection data to TFRecords file format with Example protos.

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'

    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random
import glob
import numpy as np
import tensorflow as tf
# import pdb
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
import cv2

# TFRecords convertion parameters.
RANDOM_SEED = 4242
# Yilong: Modify the samples per file
SAMPLES_PER_FILES = 200

VEHICLE_TYPE = {10.0: 'car', 20.0: 'truck'}


def get_gt_from_txt(gt_file, shape):
    "Get ground truth for a single image"
    with open(gt_file) as f:
        lines = f.read().splitlines()
    gt_info = []
    for x in lines:
        x_float = [float(item) for item in x.split(',')]
        gt_info.append(x_float)
    gt_matrix = np.array(gt_info)
    # Each line contains bounding box location and vehicle type
    gt_bboxes = gt_matrix[:, 0:4]
    gt_classes = gt_matrix[:, 4]
    return gt_bboxes, gt_classes


def _process_image(directory, name):
    """Process a image and annotation file.
    """
    # Read the image file.
    filename = directory + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'r').read()
    img = cv2.imread(filename)
    height, width, depth = img.shape
    # Get the image shape
    shape = [height, width, depth]

    # Find annotations.
    # Read  the txt groundtruth data, each line represents a bounding box
    gt_filename = os.path.join(directory, name + '_gt.txt')
    with open(gt_filename) as f:
        lines = f.read().splitlines()
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    # for each line, i.e. for each boundingbox
    for line in lines:
        line_float = [float(item) for item in line.split(',')]
        label = line_float[-1]
        labels.append(int(label))
        labels_text.append(VEHICLE_TYPE[label])
        # order of bboxes = [ymin, xmin, ymax, xmax]
        # ordero of bounding boxes in gt.txt file = [xmin, ymin, xmax, ymax]
        # print label
        bboxes.append((float(line_float[1]) / shape[0],  # ymin
                       float(line_float[0]) / shape[1],  # xmin
                       float(line_float[3]) / shape[0],  # ymax
                       float(line_float[2]) / shape[1]  # xmax
                       ))
        difficult.append(0)
        truncated.append(0)
    # print bboxes
    # print labels
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='munich_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir)
    # Get the list of images
    filenames = sorted(glob.glob(os.path.join(path, '*.jpg')))
    # Get the filenames: each element in filenames is "****.jpg"!!
    filenames = [x.split('/')[-1] for x in filenames]
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the Munich Vechile dataset!')
