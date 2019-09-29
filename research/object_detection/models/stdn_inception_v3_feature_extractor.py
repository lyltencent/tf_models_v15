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
    # feature1_tr = tf.depth_to_space(feature1, 2)
    feature2_tr = tf.depth_to_space(feature2, 2)
    feature3_tr = tf.depth_to_space(feature3, 4)

    output_features['Mixed_7a_pool'] = feature1_pool
    output_features['Mixed_7b_pool'] = feature2_pool
    output_features['Mixed_7c_pool'] = feature3_pool

    # output_features['Mixed_7a_upscale'] = feature1_tr
    output_features['Mixed_7b_upscale'] = feature2_tr
    output_features['Mixed_7c_upscale'] = feature3_tr

    # Also add the original spatial resolution.
    output_features['Mixed_7a'] = feature1

    return output_features


def combine_and_scale_transfer_module_v1(features, combine_mode=1):
    output_features = {}
    #  35 x 35 x 288.
    end_point = 'Mixed_5d'
    features_5d = features[end_point]

    #  17 x 17 x 768.
    end_point = 'Mixed_6e'
    features_6e = features[end_point]
    tmp = slim.max_pool2d(features_5d, [3, 3], scope='MaxPool_0a_3x3')
    # Combine 6e and 5d to simulate "DenseNet"
    features_6e_and_5d = tf.concat(axis=3, values=[features_6e, tmp])

    # 8 x 8 x 1280
    feature1 = features['Mixed_7a']
    # 8 x 8 x 2048
    feature2 = features['Mixed_7b']
    # 8 x 8 x 2048
    feature3 = features['Mixed_7c']

    # 1st feature map:
    #  35 x 35 x 288.
    output_features['Mixed_5d'] = features_5d

    # 2nd and 3rd feature map
    if combine_mode:
        tmp = slim.max_pool2d(features_6e_and_5d, [3, 3], scope='MaxPool_0a_3x3')
        # Combines 5d, 6e and 7c
        features_combine_all = tf.concat(axis=3, values=[feature3, tmp])
        #  17 x 17 x 768
        output_features['Mixed_6e_5d'] = features_6e_and_5d
        # 8 x 8 x 2048
        output_features['Mixed_7c_6e_5d'] = features_combine_all
    else:
        output_features['Mixed_6e'] = features_6e
        # Only combine 5d and 6e
        features_combine_all = slim.max_pool2d(features_6e_and_5d, [3, 3], scope='MaxPool_0a_3x3')
        # 8 x 8 x 2048
        output_features['Mixed_7c_6e_5d'] = features_combine_all

    # 4th, 5th, and 6th feature maps
    # output_features['Mixed_7a_upscale'] = feature1_tr
    output_features['Mixed_7b_upscale'] = tf.depth_to_space(feature2, 2)
    output_features['Mixed_7c_upscale'] = tf.depth_to_space(feature3, 4)

    # Also add the original spatial resolution.
    output_features['Mixed_7a'] = feature1

    return output_features


def combine_and_scale_transfer_module_v2(features):
    output_features = {}
    #  17 x 17 x 768.
    end_point = 'Mixed_6e'
    features_6e = features[end_point]
    # 8 x 8 x 768
    feature_6e_pool = slim.max_pool2d(features_6e, [3,3], scope='MaxPool_3x3')

    # Second, create new feature maps.
    # 8 x 8 x 1280
    feature1 = features['Mixed_7a']
    # 8 x 8 x 2048
    feature2 = features['Mixed_7b']
    # 8 x 8 x 2048
    feature3 = features['Mixed_7c']
    # 8 x 8 x 2048
    feature4 = tf.concat(axis=3, values=[feature_6e_pool, feature1])
    # 8 x 8 x (2048 + 768) = 8 x 8 x 2816
    feature5 = tf.concat(axis=3, values=[feature_6e_pool, feature2])
    # 8 x 8 x (2048 + 768) = 8 x 8 x 2816
    feature6 = tf.concat(axis=3, values=[feature_6e_pool, feature3])

    output_features['Mixed_7a_pool'] = slim.avg_pool2d(feature1, [8,8], scope='AvgPool_8x8')
    output_features['Mixed_7b_pool'] = slim.avg_pool2d(feature2, [4,4], scope='AvgPool_4x4')
    output_features['Mixed_7c_pool'] = slim.avg_pool2d(feature3, [2,2], scope='AvgPool_2x2')
    output_features['Mixed_7a_6e'] = feature4
    output_features['Mixed_7b_6e_upscale'] = tf.depth_to_space(feature5, 2)
    output_features['Mixed_7c_6e_upsacle'] = tf.depth_to_space(feature6, 4)

    return output_features


def scale_transfer_module_v2(features):
    """
    STM based on last 6 layers of inception_v3 net:

    mixed_17x17x768c  | Mixed_6c
    mixed_17x17x768d  | Mixed_6d
    mixed_17x17x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c

    :param features:
    :return:
    """
    output_features = {}
    feature1 = slim.avg_pool2d(features['Mixed_6c'], [17, 17], stride=17, scope='AvgPool_6c_17x17')
    # 1 x 1 x 768
    output_features['Mixed_6c_pool'] = feature1
    feature2 = slim.avg_pool2d(features['Mixed_6d'], [8, 8], stride=8, scope='AvgPool_6d_8x8')
    # 2 x 2 x 768
    output_features['Mixed_6d_pool'] = feature2
    feature3 = slim.avg_pool2d(features['Mixed_6e'], [4, 4], stride=4, scope='AvgPool_6e_4x4')
    # 4 x 4 x 768
    output_features['Mixed_6e_pool'] = feature3
    # 8 x 8 x 1280
    feature4 = tf.identity(features['Mixed_7a'])
    output_features['Mixed_7a'] = feature4
    # 16 x 16 x 512
    feature5 = tf.depth_to_space(features['Mixed_7b'], 2)
    output_features['Mixed_7b_upscale'] = feature5
    # 32 x 32 x 128
    feature6 = tf.depth_to_space(features['Mixed_7c'], 4)
    output_features['Mixed_7c_upscale'] = feature6

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
        # Make sure that input is in correct format with rank 4.
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
                    # 1. STDN_vesion_1
                    # image_features = scale_transfer_module(image_features)
                    # 2. STDN version + combine mode
                    image_features = combine_and_scale_transfer_module_v1(image_features, combine_mode=0)
                    # 3. STDN vesion2: result is worse than (2)
                    # image_features = scale_transfer_module_v2(image_features)
                    # 4. STDN combine_tansfer_v2
                    # image_features = combine_and_scale_transfer_module_v2(image_features)

        # return a list of feature maps
        return image_features.values()
