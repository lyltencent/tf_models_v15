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

"""STDN FeatureExtractor for InceptionV3 features."""
import tensorflow as tf

from object_detection.meta_architectures import stdn_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from nets import inception_v3

slim = tf.contrib.slim


def scale_transfer_module(features):
    output_features = {}

    # 8 x 8 x 1280
    feature1 = features['Mixed_7a']
    # 8 x 8 x 2048
    feature2 = features['Mixed_7b']
    # 8 x 8 x 2048
    feature3 = features['Mixed_7c']

    # pooling of the existing features
    feature1_pool = slim.avg_pool2d(feature1, [8, 8], stride=8, scope='AvgPool_7a_8x8')
    feature2_pool = slim.avg_pool2d(feature2, [4, 4], stride=4, scope='AvgPool_7b_4x4')
    feature3_pool = slim.avg_pool2d(feature3, [2, 2], stride=2, scope='AvgPool_7c_2x2')

    # scale transfer of the features
    feature1_tr = tf.depth_to_space(feature1, 2)
    feature2_tr = tf.depth_to_space(feature2, 2)
    feature3_tr = tf.depth_to_space(feature3, 4)

    output_features['Mixed_7a_pool'] = feature1_pool
    output_features['Mixed_7b_pool'] = feature2_pool
    output_features['Mixed_7c_pool'] = feature3_pool

    output_features['Mixed_7a_upscale'] = feature1_tr
    output_features['Mixed_7b_upscale'] = feature2_tr
    output_features['Mixed_7c_upscale'] = feature3_tr

    # Also add the original spatial resolution.
    output_features['Mixed_7c'] = feature3

    return output_features


class STDNInceptionV3FeatureExtractor(stdn_meta_arch.STDNFeatureExtractor):
    """STDN Feature Extractor using InceptionV3 features."""

    def __init__(self,
                 is_training,
                 depth_multiplier,
                 min_depth,
                 pad_to_multiple,
                 conv_hyperparams,
                 batch_norm_trainable=True,
                 reuse_weights=None):
        """InceptionV3 Feature Extractor for STDN Models.
    
        Args:
          is_training: whether the network is in training mode.
          depth_multiplier: float depth multiplier for feature extractor.
          min_depth: minimum feature extractor depth.
          pad_to_multiple: the nearest multiple to zero pad the input height and
            width dimensions to.
          conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
          batch_norm_trainable: Whether to update batch norm parameters during
            training or not. When training with a small batch size
            (e.g. 1), it is desirable to disable batch norm update and use
            pretrained batch norm params.
          reuse_weights: Whether to reuse variables. Default is None.
        """
        super(STDNInceptionV3FeatureExtractor, self).__init__(
            is_training, depth_multiplier, min_depth, pad_to_multiple,
            conv_hyperparams, batch_norm_trainable, reuse_weights)

    def preprocess(self, resized_inputs):
        """STDN preprocessing.
    
        Maps pixel values to the range [-1, 1].
    
        Args:
          resized_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
    
        Returns:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
        """
        return (2.0 / 255.0) * resized_inputs - 1.0

    def extract_features(self, preprocessed_inputs):
        """Extract features from preprocessed inputs.
    
        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.
    
        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        preprocessed_inputs.get_shape().assert_has_rank(4)
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                           tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        # feature_map_layout = {
        #     'from_layer': ['Mixed_7a_pool', 'Mixed_7b_pool', 'Mixed_7c_pool', '', '', ''],
        #     'layer_depth': [-1, -1, -1, 512, 256, 128],
        # }

        with tf.control_dependencies([shape_assert]):
            with slim.arg_scope(self._conv_hyperparams):
                with tf.variable_scope('InceptionV3',
                                       reuse=self._reuse_weights) as scope:
                    _, image_features = inception_v3.inception_v3_base(
                        ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                        final_endpoint='Mixed_7c',
                        min_depth=self._min_depth,
                        depth_multiplier=self._depth_multiplier,
                        scope=scope)

                    # Scale Transfer Module
                    image_features = scale_transfer_module(image_features)
        # return a list of feature maps
        return image_features.values()
