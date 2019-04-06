import os
import glob
import argparse
import math
import numpy, cv2
import random
# functions and tools to process the original munich dataset

DATASET_ROOT = '/Users/Forbest/Documents/Images/Aerial_images/MunichDatasetVehicleDetection-2015-old'
SET_NAME = 'Train'
SUB_IMG_WID, SUB_IMG_HEI = 300, 300
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


def rotate_bbox_to_bbox(center, size, angle):
    """
    Convert the rotated bounding box annotation (center, size, angle) to VOC bounding box annotation (x1, y1, x2, y2)
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
    x1, y1, x2, y2 = min(X_cc), min(Y_cc), max(X_cc), max(Y_cc)
    return x1, y1, x2, y2


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

    Note that the ground truth bounding box is annotated as [x1, y1, x2, y2, vehicle_type].

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
                x1, y1, x2, y2 = rotate_bbox_to_bbox(center, size, angle)

                file_name = os.path.join(img_path, '{}_bbox.gt'.format(img_name))
                with open(file_name, 'a+') as f:
                    write_str = '{} {} {} {} {} \n'.format(x1, y1, x2, y2, vehicle_type)
                    f.write(write_str)


def crop_images_and_generate_groundtruth(img_path, img_name, save_path):
    """
    Crop a sub_image from the original image, and generate ground truth map if there are targets inside.

    Note that the original labels are in the order of [x1, y1, x2, y2]

    """
    gt_file = os.path.join(img_path, '{}_bbox.gt'.format(img_name))
    # Coordinates in gt_file is in the order of [x1, y1, x2, y2]
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
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def select_subimage_anno(w, h):
        select_box = []
        for box in target_annos:
            x1, y1, x2, y2, cat= box
            x11, y11 = x1 - w, y1 - h
            x22, y22 = x2 - w, y2 - h
            gx1, gy1 = w, h
            gx2, gy2 = w + SUB_IMG_WID, h + SUB_IMG_HEI
            overlap_area = twoboxes_overlap(box, [gx1, gy1, gx2, gy2])
            if overlap_area <= 0:
                continue
            # bbox = [xmin, ymin, xmax, ymax]
            new_box = [max(0, x11), max(0, min(y11, SUB_IMG_HEI)), max(0, x22), min(y22, SUB_IMG_HEI), cat]
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
            clone_image = image_data.copy()
            sub_image = clone_image[h:h+SUB_IMG_HEI, w:w+SUB_IMG_WID]
            # select_annos: [x1, y1, x2, y2]
            select_annos = select_subimage_anno(w, h)
            if len(select_annos) == 0:
                continue

            print(len(select_annos), select_annos)
            if DRAW_BBOX:
                for box in select_annos:
                    x1, y1, x2, y2, cat = box
                    # Use two points to draw in cv2.rectangle
                    cv2.rectangle(sub_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            save_image_path = os.path.join(save_path, "{}_{}_{}.jpg".format(img_name, w, h))
            save_anno_path = os.path.join(save_path, "{}_{}_{}_gt.txt".format(img_name, w, h))
            cv2.imwrite(save_image_path, sub_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            with open(save_anno_path, "w") as writer:
                for box in select_annos:
                    writer.write(",".join(map(str, box)) + "\n")


def convert_image_ground_truth(image_path):
    img_names = glob.glob(os.path.join(image_path, '*.JPG'))
    img_names = [os.path.splitext(os.path.basename(x))[0] for x in img_names]
    for img_name in img_names:
        convert_single_image_vehicle_info(image_path, img_name)


def split_train_val_set(crop_image_path, ratio=0.8):
    """
    crop_image_path: path containing the crop images and their ground truth files
    :param crop_image_path:
    :param ratio: ratio of training examples from training data.
    :return:
    """
    filenames = glob.glob(os.path.join(crop_image_path, '*.jpg'))
    filenames.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    # 80% of the training data is used for traning, the rest 15% used for validation
    split_1 = int(ratio * len(filenames))
    train_filenames = filenames[:split_1]
    val_filenames = filenames[split_1:]

    # Save the file name into two txt files: 'train.txt' and 'val.txt'
    train_filenames = [os.path.splitext(os.path.basename(x))[0] for x in train_filenames]
    val_filenames = [os.path.splitext(os.path.basename(x))[0] for x in val_filenames]
    train_file = os.path.join(crop_image_path, 'train.txt')
    with open(train_file, 'w') as f:
        f.write("\n".join(train_filenames))

    val_file = os.path.join(crop_image_path, 'val.txt')
    with open(val_file, 'w') as f:
        f.write("\n".join(val_filenames))


def get_test_txt(crop_image_path):
    test_filenames = glob.glob(os.path.join(crop_image_path, '*.jpg'))
    test_filenames.sort()
    test_filenames = [os.path.splitext(os.path.basename(x))[0] for x in test_filenames]
    test_file = os.path.join(crop_image_path, 'test.txt')
    with open(test_file, 'w') as f:
        f.write("\n".join(test_filenames))




if __name__ == '__main__':
    # Load arguments.
    parser = argparse.ArgumentParser(description='Tools to read and create ground for Munich dataset')
    parser.add_argument('-i', '--dataset_root', type=str, default=DATASET_ROOT)
    parser.add_argument('-s', '--set_name', type=str, default=SET_NAME)
    parser.add_argument('-o', '--save_path', type=str, default='')
    parser.add_argument('--draw_bbox', action='store_true', dest='DRAW_BBOX', required=False, help='DRAW bbox on croped images or not')
    args = parser.parse_args()
    dataset_root = args.dataset_root
    set_name = args.set_name
    save_path = args.save_path
    DRAW_BBOX = args.DRAW_BBOX

    num_vehicles = get_number_of_vechiles(dataset_root, set_name)
    print ('Number of vehicles in {} is {}'.format(set_name, num_vehicles))
    # Generate ground truth for images by combining different samp files.
    convert_image_ground_truth(os.path.join(dataset_root, set_name))
    # For each image, crop sub images and generate corresponding groundtruth
    img_names = glob.glob(os.path.join(os.path.join(dataset_root, set_name), '*.JPG'))
    img_names = [os.path.splitext(os.path.basename(x))[0] for x in img_names]
    import pdb; pdb.set_trace()
    print('Cropping images into training set \n')

    if set_name.lower() == 'train':
        SUB_OVERLAP = 80
    else
        SUB_OVERLAP = 10

    for image_name in img_names:
        crop_images_and_generate_groundtruth(os.path.join(dataset_root, set_name), img_name=image_name, save_path=save_path)
    # print('Generate train.txt and val.txt. \n')
    if set_name.lower() == 'train':
        split_train_val_set(os.path.join(dataset_root, 'Train_crop'), 0.85)
    if set_name.lower() == 'test':
        get_test_txt(os.path.join(dataset_root, 'Test_crop'))
