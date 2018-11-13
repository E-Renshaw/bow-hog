""Test BOW model"""

import numpy as np
import cv2
from utils import *

# load dictionary and SVM from training
try:
    dictionary = np.load(params.BOW_DICT_PATH)
    svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
except:
    print("Missing files - dictionary and/or SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# load test data
print("Loading test data as a batch ...")

paths = [params.DATA_testing_path_neg, params.DATA_testing_path_pos]
use_centre_weighting = [False, False]
class_names = params.DATA_CLASS_NAMES
imgs_data = load_images(paths, class_names, [0,0], use_centre_weighting)

print("Computing descriptors...")
start = cv2.getTickCount()
[img_data.compute_bow_descriptors() for img_data in imgs_data]
print_duration(start)

print("Generating histograms...")
start = cv2.getTickCount()
[img_data.generate_bow_hist(dictionary) for img_data in imgs_data]
print_duration(start)

# get the example/sample bow histograms and class labels
samples, class_labels = get_bow_histograms(imgs_data), get_class_labels(imgs_data)

# classify whole set
print("Performing batch SVM classification over all data  ...")
results = svm.predict(samples)
output = results[1].ravel()

# get error
error = ((np.absolute(class_labels.ravel() - output).sum()) / float(output.shape[0]))
print("Successfully trained SVM with {}% testing set error".format(round(error * 100,2)))
print("-- meaining the SVM got {}% of the testing examples correct!".format(round((1.0 - error) * 100,2)))
