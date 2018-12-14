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

"""Test for create_munich_vehicle_tf_record.py."""

import os

import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import create_munich_vehicle_tf_record


class DictToTFExampleTest(tf.test.TestCase):

  def _assertProtoEqual(self, proto_field, expectation):
    """Helper function to assert if a proto field equals some value.

    Args:
      proto_field: The protobuf field to compare.
      expectation: The expected value of the protobuf field.
    """
    proto_list = [p for p in proto_field]
    self.assertListEqual(proto_list, expectation)

  def test_dict_to_tf_example(self):
    image_path = '/Documents/tf_models_v15/research/functional_test'
    image_file_name = 'tmp_img.jpg'

    label_map_dict = {
        'background': 0,
        'car': 10,
        'truck': 20,
    }
    example = create_munich_vehicle_tf_record.dict_to_tf_example(
        split_data_dir=image_path, name='tmp_img')
    self._assertProtoEqual(
        example.features.feature['image/height'].int64_list.value, [600])
    self._assertProtoEqual(
        example.features.feature['image/width'].int64_list.value, [700])
    self._assertProtoEqual(
        example.features.feature['image/filename'].bytes_list.value,
        [image_file_name])
    self._assertProtoEqual(
        example.features.feature['image/source_id'].bytes_list.value,
        [image_file_name])
    self._assertProtoEqual(
        example.features.feature['image/format'].bytes_list.value, ['jpeg'])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmin'].float_list.value,
        [0.5, 0.5])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymin'].float_list.value,
        [0.5, 0.5])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/xmax'].float_list.value,
        [1.0, 1.0])
    self._assertProtoEqual(
        example.features.feature['image/object/bbox/ymax'].float_list.value,
        [1.0, 1.0])
    self._assertProtoEqual(
        example.features.feature['image/object/class/text'].bytes_list.value,
        ['car', 'truck'])
    self._assertProtoEqual(
        example.features.feature['image/object/class/label'].int64_list.value,
        [1, 2])
    self._assertProtoEqual(
        example.features.feature['image/object/difficult'].int64_list.value,
        [0, 0])
    self._assertProtoEqual(
        example.features.feature['image/object/truncated'].int64_list.value,
        [0, 0])
    self._assertProtoEqual(
        example.features.feature['image/object/view'].bytes_list.value, ['none', 'none'])


if __name__ == '__main__':
  tf.test.main()
