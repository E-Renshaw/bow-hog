"""Test HOG model"""

import numpy as np
import cv2
from utils import *

# load SVM
try:
    svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
except:
    print("Missing files  SVM")
    print("-- have you performed training to produce this file ?")
    exit()

# load test data
print("Loading test data as a batch ...")

paths = [params.DATA_testing_path_neg, params.DATA_testing_path_pos]
use_centre_weighting = [False, True]
class_names = params.DATA_CLASS_NAMES
imgs_data = load_images(paths, class_names, [0,0], use_centre_weighting)

print("Computing HOG descriptors...")
start = cv2.getTickCount()
[img_data.compute_hog_descriptor() for img_data in imgs_data]
print_duration(start)

# get the example/sample HOG descriptors and class labels
samples, class_labels = get_hog_descriptors(imgs_data), get_class_labels(imgs_data)

# classify over whole dataset
print("Performing batch SVM classification over all data  ...")
results = svm.predict(samples)
output = results[1].ravel()

# get error
error = ((np.absolute(class_labels.ravel() - output).sum()) / float(output.shape[0]))
print("Successfully trained SVM with {}% testing set error".format(round(error * 100,2)))
print("-- meaining the SVM got {}% of the testing examples correct!".format(round((1.0 - error) * 100,2)))
