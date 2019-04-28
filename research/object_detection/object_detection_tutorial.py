import numpy as np
import os
import tensorflow as tf
import glob
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
import numpy as np

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

flags = tf.app.flags

flags.DEFINE_string('path_to_ckpt', '',
                    'Path to a frozen detection graph')
flags.DEFINE_string('path_to_labels', '',
                    'Path to label pb txt file')
flags.DEFINE_integer('num_classes', 0,
                     'Number of classes to for detection')
flags.DEFINE_string('path_to_test_imgs', '',
                    'Path to test images.')
flags.DEFINE_string('saved_path', '',
                    'Path to save inference output.')

FLAGS = flags.FLAGS


def main(_):
    PATH_TO_CKPT = FLAGS.path_to_ckpt
    NUM_CLASSES = FLAGS.num_classes
    PATH_TO_LABELS = FLAGS.path_to_labels
    PATH_TO_TEST_IMAGES_DIR = FLAGS.path_to_test_imgs
    RESULT_VIS_PATH = FLAGS.saved_path


    # Step 1: Load a frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    # Step 2: Load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Step 3: Get the list of image names
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))
    # TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image_{}.jpg'.format(i)) for i in range(1, 11)]
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    if not os.path.exists(RESULT_VIS_PATH):
        os.makedirs(RESULT_VIS_PATH)
    inferece_time = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                # the array based representation of the image will be used in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # expand dimensions sinc eth model expects image to have shape [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                start_time = time.time()
                (boxes, scores, classes, num) =  sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )
                elapse_time = time.time() - start_time
                inferece_time.append(elapse_time)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3)
                plt.imsave(os.path.join(RESULT_VIS_PATH, os.path.basename(image_path)), image_np)
    print ("Time elapsed for each frame = {}".format(inferece_time))
    print ("Speed of testing is {} seconds/frame".format(np.mean(inferece_time[1:])))


if __name__ == '__main__':
    tf.app.run()
