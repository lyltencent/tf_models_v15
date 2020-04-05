import tensorflow as tf

from object_detection.meta_architectures import stdn_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from nets import densenet
slim = tf.contrib.slim


def scale_transfer_module_densent_169(image_features):
    output_features = {}
    feature1 = image_features['FeatureExtractor/densenet169/densenet169/dense_block4/conv_block5']
    feature2 = image_features['FeatureExtractor/densenet169/densenet169/dense_block4/conv_block10']
    feature3 = image_features['FeatureExtractor/densenet169/densenet169/dense_block4/conv_block15']
    feature4 = image_features['FeatureExtractor/densenet169/densenet169/dense_block4/conv_block20']
    feature5 = image_features['FeatureExtractor/densenet169/densenet169/dense_block4/conv_block25']
    feature6 = image_features['FeatureExtractor/densenet169/densenet169/dense_block4/conv_block32']

    output_features['DB4_concat5_pooling'] = slim.avg_pool2d(feature1, [9, 9], scope='AvgPool')
    output_features['DB4_concat10_pooling'] = slim.avg_pool2d(feature2, [3, 3], scope='AvgPool_db4_concat10')
    output_features['DB4_concat15_pooling'] = slim.avg_pool2d(feature3, [2, 2], scope='AvgPool_db4_concat15')
    output_features['DB4_concat20_idendity'] = tf.identity(feature4)
    output_features['DB4_concat25_scale_transfer'] = tf.depth_to_space(feature5, 2)
    output_features['DB4_concat32_scale_transfer'] = tf.depth_to_space(feature6, 4)

    return output_features


class STDNDenseNet169FeatureExtractor(stdn_meta_arch.STDNFeatureExtractor):
    """STDN Feature Extractor using Pretrained DenseNet_121 checkpoint."""
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
        super(STDNDenseNet169FeatureExtractor, self).__init__(
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
                with tf.variable_scope('densenet169',
                                       reuse=self._reuse_weights) as scope:
                    _, image_features = densenet.densenet169(
                            preprocessed_inputs,
                            num_classes=1000,
                            data_format='NHWC',
                            is_training=True,
                            reuse=self._reuse_weights)
                # Insert scale transfer module
                image_features = scale_transfer_module_densent_169(image_features)

        # return a list of feature maps
        return image_features.values()
