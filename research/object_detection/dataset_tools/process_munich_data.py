import os
import glob
import argparse
import math
import numpy
# functions and tools to process the original munich dataset

DATASET_ROOT = '/Users/Forbest/Documents/Images/Aerial_images/MunichDatasetVehicleDetection-2015-old'
SET_NAME = 'Train'


def get_number_of_vechiles(dataset_root, set_name):
    """
    Get the nubmer of vehicles in the samp files.  sampel samp files:

    @CATEGORY:GENERAL
    @IMAGE:2012-04-26-Muenchen-Tunnel_4K0G0010.JPG
    # format: id type center.x center.y size.width size.height angle
    0 30 1319 2338 35 11 56.451578
    1 30 1337 2350 42 14 57.817368
    2 30 224 3556 61 20 136.967797

     => number of vechiles = 3

    :param datset_root_path:
    :param set_name:
    :return:
    """
    samp_list = glob.glob(os.path.join(dataset_root, set_name, '*.samp'))
    num_vehicles = 0
    for cur_samp in samp_list:
        lines = open(cur_samp).readlines()
        cur_samp_num_vechiles = int(float(lines[-1].split()[0])) + 1
        num_vehicles = num_vehicles + cur_samp_num_vechiles
    return num_vehicles


def rotate_bbox_to_bbox (center, size, angle):
    """
    Convert the rotated bounding box annotation (center, size, angle) to VOC bounding box annotation (x1, x2, y1, y2)
    :param center:
    :param size:
    :param angle:
    :return:
    """
    width = size[0]
    height = size[1]
    X_rec = numpy.array([center[0] - width, center[0] - width, center[0] + width, center[0] + width, center[0] - width])
    Y_rec = numpy.array([center[1] - height, center[1] + height, center[1] + height, center[1] - height, center[1] - height])
    # import pdb; pdb.set_trace()
    X_c = X_rec - center[0]
    Y_c = Y_rec - center[1]
    current_angle_d = -angle
    current_angle_radian = math.radians(current_angle_d)
    X_cc = math.cos(current_angle_radian)*X_c - math.sin(current_angle_radian)*Y_c
    Y_cc = math.sin(current_angle_radian)*X_c + math.cos(current_angle_radian)*Y_c
    X_cc = X_cc + center[0]
    Y_cc = Y_cc + center[1]
    # Create bounding box location in VOC 2007 format
    x1, x2, y1, y2 = min(X_cc), max(X_cc), min(Y_cc), max(Y_cc)
    return x1, x2, y1, y2


def convert_single_image_vehicle_info(img_path, img_name):
    """
    For each image, there exist several *.samp files containing the object ground truth locations.
    Each ground truth is in the format of:
    ------------------------------------------------------------------
    id    type   center.x   center.y   size.width    size.height    angle
    0      22     4582       1636        47            23           -147.673860
    ------------------------------------------------------------------
    For vehicle detection, we only consider 7 types of *samp files, with corresponding type id:
    _bus.samp (30),           _cam.samp (20),
    _pkw_trail.samp (11),     _pkw.samp (10, 16),
    _truck.samp (22) ,        _truck_trail.samp(23),
    _van_trail (17)

    """
    vehicle_types = {'bus': 30, 'cam': 20, 'pkw_trail': 11, 'pkw': 10, 'truck': 22, 'truck_trail': 23, 'van_trail': 17}

    for type_i in vehicle_types.keys():
        type_i_samp_file = os.path.join(img_path, '{}_{}.samp'.format(img_name, type_i))
        # If there is the correspoing type in the image, get the values
        if os.path.exists(type_i_samp_file):
            lines = open(type_i_samp_file).readlines()
            # Get the starting_line_number for storing ground truth
            for index, line in enumerate(lines):
                if line.startswith('# format'):
                    starting_line_number = index + 1
                    break
            for idx in range(starting_line_number, len(lines)):
                line_content = [float(x) for x in lines[idx].split()]
                # Get the bounding box center location
                center = line_content[2:4]
                size = line_content[4:6]
                vehicle_type = type_i
                # get the angle in radians
                angle = line_content[-1]
                x1, x2, y1, y2 = rotate_bbox_to_bbox(center, size, angle)

                file_name = os.path.join(img_path, '{}_bbox.gt'.format(img_name))
                with open(file_name, 'a+') as f:
                    write_str = '{} {} {} {} {} \n'.format(x1, x2, y1, y2, vehicle_type)
                    f.write(write_str)


def convert_image_ground_truth(image_path):
    img_names = glob.glob(os.path.join(image_path, '*.JPG'))
    img_names = [os.path.splitext(os.path.basename(x))[0] for x in img_names]
    for img_name in img_names:
        convert_single_image_vehicle_info(image_path, img_name)


if __name__ == '__main__':
    # Load arguments.
    parser = argparse.ArgumentParser(description='Tools to read and create ground for Munich dataset')
    parser.add_argument('-i', '--dataset_root', type=str, default=DATASET_ROOT)
    parser.add_argument('-s', '--set_name', type=str, default=SET_NAME)
    args = parser.parse_args()
    dataset_root = args.dataset_root
    set_name = args.set_name

    # num_vehicles = get_number_of_vechiles(dataset_root, set_name)
    # print ('Number of vehicles in {} is {}'.format(set_name, num_vehicles))
    convert_image_ground_truth(dataset_root)
