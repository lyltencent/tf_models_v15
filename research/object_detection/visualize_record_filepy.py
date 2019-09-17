
import functools
import os
import tensorflow as tf
import glob
from object_detection.data_decoders import tf_example_decoder
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from object_detection.utils import dataset_util

parallel_reader = tf.contrib.slim.parallel_reader

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags

flags.DEFINE_string('record_file_path', '',
                    'Direcotory of the record file')
flags.DEFINE_string('vis_dir', '',
                    'Directory to write the images to')
flags.DEFINE_string('label_map_path', '',
                    'Direcotry of label map file')

FLAGS = flags.FLAGS


def main(unused_argv):
    assert FLAGS.record_file_path, '`record_file_path` is missing.'
    assert FLAGS.vis_dir, '`vis_dir` is missing.'
    tf.gfile.MakeDirs(FLAGS.vis_dir)
    filename = glob.glob(FLAGS.record_file_path)
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer(filename)
    _, serialized_example = reader.read(filename_queue)
    feature_set = {
        'image/height': tf.train.Int64List(),
        'image/width': tf.train.Int64List(),
        # 'image/filename': tf.train.BytesList(),
        'image/source_id': tf.train.BytesList(),
        'image/key/sha256': tf.train.BytesList(),
        # 'image/encoded': tf.train.BytesList(),
        # 'image/format': tf.train.BytesList(),
        'image/object/bbox/xmin': tf.train.FloatList(),
        'image/object/bbox/xmax': tf.train.FloatList(),
        'image/object/bbox/ymin': tf.train.FloatList(),
        'image/object/bbox/ymax': tf.train.FloatList(),
        # 'image/object/class/text': tf.train.BytesList(),
        'image/object/class/label': tf.train.Int64List(),
        'image/object/difficult': tf.train.Int64List(),
        'image/object/truncated': tf.train.Int64List(),
        # 'image/object/view': tf.train.BytesList(),
    }
    import pdb; pdb.set_trace()
    features = tf.parse_single_example(serialized_example, features=feature_set)


    # groundtruth = None
    # if not ignore_groundtruth:
    #     groundtruth = {
    #         fields.InputDataFields.groundtruth_boxes:
    #             input_dict[fields.InputDataFields.groundtruth_boxes],
    #         fields.InputDataFields.groundtruth_classes:
    #             input_dict[fields.InputDataFields.groundtruth_classes],
    #         fields.InputDataFields.groundtruth_area:
    #             input_dict[fields.InputDataFields.groundtruth_area],
    #         fields.InputDataFields.groundtruth_is_crowd:
    #             input_dict[fields.InputDataFields.groundtruth_is_crowd],
    #         fields.InputDataFields.groundtruth_difficult:
    #             input_dict[fields.InputDataFields.groundtruth_difficult]
    #     }
    #     if fields.InputDataFields.groundtruth_group_of in input_dict:
    #         groundtruth[fields.InputDataFields.groundtruth_group_of] = (
    #             input_dict[fields.InputDataFields.groundtruth_group_of])
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # input_dict = sess.run(input_dict)
    # sess.close()

if __name__ == '__main__':
    tf.app.run()
