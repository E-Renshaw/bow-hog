"""Detect based on BOW model"""

import cv2
import os
import numpy as np
import math
import params
from utils import *
from sliding_window import *

directory_to_cycle =  # path to target datset

try:
    dictionary = np.load(params.BOW_DICT_PATH)
    svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
except:
    print("Missing files - dictionary and/or SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# checks
print("dictionary size : {}".format(dictionary.shape))
print("svm size : {}".format(len(svm.getSupportVectors())))
print("svm var count : {}".format(svm.getVarCount()))

# go through whole directory
for filename in sorted(os.listdir(directory_to_cycle)):
    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename))

        img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)
        output_img = img.copy() # make a copy for output

        # init different image scales
        current_scale = -1
        detections = []
        rescaling_factor = 1.25
        for resized in pyramid(img, scale=rescaling_factor):
            if current_scale == -1:
                current_scale = 1
            else:
                current_scale /= rescaling_factor # scale down each time
            rect_img = resized.copy()

            # show progress
            if (show_scan_window_process):
                cv2.imshow('current scale',rect_img)
                cv2.waitKey(10)
            # loop over window for each layer in image
            window_size = params.DATA_WINDOW_SIZE
            step = math.floor(resized.shape[0] / 16)

            if step > 0:
                for (x, y, window) in sliding_window(resized, window_size, step_size=step):
                    # show progress
                    if (show_scan_window_process):
                        cv2.imshow('current window',window)
                        key = cv2.waitKey(10)
                    # get BOW descriptors
                    img_data = ImageData(window)
                    img_data.compute_bow_descriptors()

                    # classify by constructing BOW histogram and passing through SVM
                    if img_data.bow_descriptors is not None:
                        img_data.generate_bow_hist(dictionary)
                        print("detecting with SVM ...")
                        retval, [result] = svm.predict(np.float32([img_data.bow_histogram]))
                        print(result)

                        # record any detections
                        if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:
                            # in form (x1, y1) (x2,y2) pair
                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])
                            # show progress
                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                                cv2.imshow('current scale',rect_img)
                                cv2.waitKey(10)

                            rect *= (1.0 / current_scale)
                            detections.append(rect)

        # remove overlapping boxes
        detections = non_max_suppression_fast(np.int32(detections), 0.4)

        # show detections
        for rect in detections:
            cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        cv2.imshow('detected objects',output_img)
        key = cv2.waitKey(40)
        if (key == ord('x')):
            break

cv2.destroyAllWindows()
