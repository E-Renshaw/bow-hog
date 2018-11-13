# Object Detection using Bag of Words and Histogram of Oriented Gradients

This repo contains implementations of two methods for object detection: Bag of Words and Histogram of Oriented Gradients.

All implementations require Python 3 and OpenCV 3.4.

### Usage

1. Edit the master_path_to_dataset variable in params.py to the location you have placed your training/testing files
1. Run {hog|bow}_train.py
1. Run {hog|bow}_test.py
1. Edit the directory_to_cycle variable in {hog|bow}_detector.py to the target dataset
1. Run {hog|bow}_detector.py

---

Note that here we use an SVM for classification, so it takes a long time to train!
