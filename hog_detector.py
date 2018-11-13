""Detect using HOG"""

import cv2
import os
import numpy as np
import math
import params
from utils import *
from sliding_window import *

directory_to_cycle = # change this to desired filepath

show_scan_window_process = True

# load SVM
try:
    svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
except:
    print("Missing files - SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# checks
print("svm size : ", len(svm.getSupportVectors()))
print("svm var count : ", svm.getVarCount())

# go through all images in directory
for filename in sorted(os.listdir(directory_to_cycle)):
    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename))

        img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)
        # make a copy for output
        output_img = img.copy()

        # init different image scales
        current_scale = -1
        detections = []
        rescaling_factor = 1.25

        for resized in pyramid(img, scale=rescaling_factor):
            if current_scale == -1:
                current_scale = 1
            else:
                current_scale /= rescaling_factor
            rect_img = resized.copy()
            # progress
            if (show_scan_window_process):
                cv2.imshow('current scale',rect_img)
                cv2.waitKey(10)
            # loop over window for each layer in image
            window_size = params.DATA_WINDOW_SIZE
            step = math.floor(resized.shape[0] / 16)

            if step > 0:
                for (x, y, window) in sliding_window(resized, window_size, step_size=step):
                    # progress
                    if (show_scan_window_process):
                        cv2.imshow('current window',window)
                        key = cv2.waitKey(10)

                    # get descriptors
                    img_data = ImageData(window)
                    img_data.compute_hog_descriptor()

                    # classify by constructing HOG and passing through SVM
                    if img_data.hog_descriptor is not None:
                        print("detecting with SVM ...")
                        retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))
                        print(result)

                        # record descriptions
                        if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:
                            # in form (x1, y1) (x2,y2) pair
                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])
                            # progress
                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                                cv2.imshow('current scale',rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections.append(rect)

        # remove overlapping boxes
        detections = non_max_suppression_fast(np.int32(detections), 0.4)

        # SHOW detections
        for rect in detections:
            cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        cv2.imshow('detected objects',output_img)
        key = cv2.waitKey(200)
        if (key == ord('x')):
            break

cv2.destroyAllWindows()
