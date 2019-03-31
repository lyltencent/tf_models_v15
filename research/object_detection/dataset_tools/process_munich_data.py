import os
import glob
import argparse
import math
import numpy, cv2
# functions and tools to process the original munich dataset

DATASET_ROOT = '/Users/Forbest/Documents/Images/Aerial_images/MunichDatasetVehicleDetection-2015-old'
SET_NAME = 'Train'
SUB_IMG_WID, SUB_IMG_HEI, SUB_OVERLAP = 300, 300, 80
vehicle_types = {'bus': 30, 'cam': 20, 'pkw_trail': 11, 'pkw': 10, 'truck': 22, 'truck_trail': 23, 'van_trail': 17}


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

    Note that the ground truth bounding box is annotated as [x1, x2, y1, y2].

    """
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


def crop_images_and_generate_groundtruth(img_path, img_name, save_path):
    """
    Crop a sub_image from the original image, and generate ground truth map if there are targets inside.

    Note that the original labels are in the order of [x1, x2, y1, y2]

    """
    gt_file = os.path.join(img_path, '{}_bbox.gt'.format(img_name))
    if not os.path.exists(gt_file):
        print ('Have to genereate ground truth file for the image')
    target_annos = []
    with open(gt_file, "r") as reader:
        cnt = 0
        for line in reader:
            cnt += 1
            if cnt == 1:
                continue
            loc = [float(x) for x in line.strip().split()[:-1]]
            # loc = list(map(int, loc))
            cat_num = vehicle_types[line.strip().split()[-1]]
            loc.append(cat_num)
            target_annos.append(loc)

    def twoboxes_overlap(box1, box2):
        x1 = max(box1[0], box2[0])
        x2 = min(box1[1], box2[1])
        y1 = max(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def select_subimage_anno(w, h):
        select_box = []
        for box in target_annos:
            x1, x2, y1, y2, cat= box
            x11, y11 = x1 - w, y1 - h
            x22, y22 = x2 - w, y2 - h
            gx1, gy1 = w, h
            gx2, gy2 = w + SUB_IMG_WID, h + SUB_IMG_HEI
            overlap_area = twoboxes_overlap(box, [gx1, gx2, gy1, gy2])
            if overlap_area <= 0:
                continue
            new_box = [max(0, x11), max(0, x22), max(0, min(y11, SUB_IMG_WID)), min(y22, SUB_IMG_HEI), cat]
            # If the overlap area is more than 70% of the original ground truth, this bounding box belongs to the new sub-image.
            if overlap_area / ((x22 - x11) * (y22 - y11)) >= 0.7:
                select_box.append(new_box)
        return select_box

    image_data = cv2.imread(os.path.join(img_path, '{}.JPG'.format(img_name)))
    H, W = image_data.shape[:2]
    cnt = 0
    for h in range(0, H, SUB_IMG_HEI-SUB_OVERLAP):
        for w in range(0, W, SUB_IMG_WID-SUB_OVERLAP):
            if h + SUB_IMG_HEI >= H:
                h = H - SUB_IMG_HEI
            if w + SUB_IMG_WID >= W:
                w = W - SUB_IMG_WID
            cnt += 1
            # Get the sub_image and its ground truth locations.
            sub_image = image_data[h:h+SUB_IMG_HEI, w:w+SUB_IMG_WID]
            select_annos = select_subimage_anno(w, h)
            if len(select_annos) == 0:
                continue

            print(len(select_annos), select_annos)
            for box in select_annos:
                x1, x2, y1, y2, cat = box
                # Use two points to draw in cv2.rectangle
                cv2.rectangle(sub_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            save_image_path = os.path.join(save_path, "{}_{}_{}.jpg".format(img_name, w, h))
            save_anno_path = os.path.join(save_path, "{}_{}_{}.txt".format(img_name, w, h))
            cv2.imwrite(save_image_path, sub_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            with open(save_anno_path, "w") as writer:
                for box in select_annos:
                    writer.write(",".join(map(str, box)) + "\n")


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
    # convert_image_ground_truth(dataset_root)
    crop_images_and_generate_groundtruth(dataset_root, img_name=set_name, save_path='/Users/Forbest/Desktop/temp')
