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
from nets import inception_v2

slim = tf.contrib.slim


def combine_and_scale_transfer_module_v1(features):
    output_features = {}
    #  28 x 28 x 320
    end_point = 'Mixed_3c'
    features_3c = features[end_point]

    #  14 x 14 x 576
    end_point = 'Mixed_4e'
    features_4e = features[end_point]
    tmp = slim.max_pool2d(features_3c, [3, 3], scope='MaxPool_0a_3x3')
    # features_4e_and_3c => 14 x 14 x (320+576) = 14 x 14 x 896
    features_4e_and_3c = tf.concat(axis=3, values=[features_4e, tmp])

    # 7 x 7 x 1024
    feature1 = features['Mixed_5a']
    # 7 x 7 x 1024
    feature2 = features['Mixed_5b']
    # 7 x 7 x 1024
    feature3 = features['Mixed_5c']
    tmp = slim.max_pool2d(features_4e_and_3c, [3, 3], scope='MaxPool_0a_3x3')
    # features_combine_all => 7 x 7 x (1024+896) = 7 x 7 x 1920
    features_combine_all = tf.concat(axis=3, values=[feature3, tmp])

    #  1 x 1 x 320
    output_features['Mixed_3c'] = slim.avg_pool2d(features_3c, [28, 28], scope='AvgPool')
    #  3 x 3 x 896
    output_features['Mixed_4e_3c_pool'] = slim.avg_pool2d(features_4e_and_3c, [4, 4], scope='AvgPool_0a_4x4')
    # 7 x 7 x 1920
    output_features['Mixed_5c_4e_3c'] = features_combine_all
    # 14 x 14
    output_features['Mixed_4e_3c'] = features_4e_and_3c
    # 28 x 28
    output_features['Mixed_5c_4e_3c_upscale'] = tf.depth_to_space(features_combine_all, 4)
    # Also add the original spatial resolution.
    output_features['Mixed_5c'] = feature3

    return output_features


class STDNInceptionV2FeatureExtractor(stdn_meta_arch.STDNFeatureExtractor):
    """STDN Feature Extractor using InceptionV2 features."""

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
        super(STDNInceptionV2FeatureExtractor, self).__init__(
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
        # Make sure that input is in correct format with rank 4.
        preprocessed_inputs.get_shape().assert_has_rank(4)
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                           tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        with tf.control_dependencies([shape_assert]):
            with slim.arg_scope(self._conv_hyperparams):
                with tf.variable_scope('InceptionV3',
                                       reuse=self._reuse_weights) as scope:
                    _, image_features = inception_v2.inception_v2_base(
                        ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                        final_endpoint='Mixed_5c',
                        min_depth=self._min_depth,
                        depth_multiplier=self._depth_multiplier,
                        scope=scope)

                    # 2. STDN version + combine mode
                    image_features = combine_and_scale_transfer_module_v1(image_features)

        # return a list of feature maps
        return image_features.values()
