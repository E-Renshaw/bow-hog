"""Some parameter settings"""

import cv2
import os

# general settings

master_path_to_dataset = # the path to the dataset

# data location - training
DATA_training_path_neg = os.path.join(master_path_to_dataset,"""Edit this""")
DATA_training_path_pos = os.path.join(master_path_to_dataset,"""Edit this""")

# data location - testing
DATA_testing_path_neg = os.path.join(master_path_to_dataset,"""Edit this""")
DATA_testing_path_pos = os.path.join(master_path_to_dataset,"""Edit this""")

# size of the sliding window
DATA_WINDOW_SIZE = [64, 128]

# the maximum left/right, up/down offset to use when generating samples for training
# that are centred around the centre of the image
DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3

# number of sample patches to extract from each negative training example
DATA_training_sample_count_neg = 10

# number of sample patches to extract from each positive training example
DATA_training_sample_count_pos = 5

# class names
DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrian": 1
}

# BOW settings

BOW_SVM_PATH = "svm_bow.xml"
BOW_DICT_PATH = "bow_dictionary.npy"

BOW_dictionary_size = 512
BOW_SVM_kernel = cv2.ml.SVM_RBF
BOW_SVM_max_training_iterations = 500

BOW_clustering_iterations = 20

BOW_fixed_feature_per_image_to_use = 100

# specify the type of feature points to use
# see opencv docs for some options

# choose ORB or SIFT
BOW_use_ORB_always = False

try:
    if BOW_use_ORB_always:
        print("Forced used of ORB features, not SIFT")
        raise Exception('force use of ORB')

    DETECTOR = cv2.xfeatures2d.SIFT_create(nfeatures=BOW_fixed_feature_per_image_to_use)

    _algorithm = 0 # FLANN_INDEX_KDTREE
    _index_params = dict(algorithm=_algorithm, trees=5)
    _search_params = dict(checks=50)

except:

    DETECTOR = cv2.ORB_create(nfeatures=BOW_fixed_feature_per_image_to_use) # check these params

    #if using ORB points
    _algorithm = 6 # FLANN_INDEX_LSH
    _index_params= dict(algorithm = _algorithm,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    _search_params = dict(checks=50)

    if (not(BOW_use_ORB_always)):
        print("Falling back to using features: ", DETECTOR.__class__())
        BOW_use_ORB_always = True # set this as a flag we can check later which data type to uses

print("For BOW - features in use are: ", DETECTOR.__class__(), "(ignore for HOG)")

# based on choice and availability of feature points, set up KD-tree matcher
MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)

# HOG settings

HOG_SVM_PATH = "svm_hog.xml"

HOG_SVM_kernel = cv2.ml.SVM_LINEAR # see opencv manual for other options
HOG_SVM_max_training_iterations = 500
