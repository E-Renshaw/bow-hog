"""Train BOW model"""

import cv2
from utils import *

def generate_dictionary(imgs_data, dictionary_size):
    # get descriptors
    desc = stack_array([img_data.bow_descriptors for img_data in imgs_data])
    desc = np.float32(desc)

    # clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, params.BOW_clustering_iterations, 0.01)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, dictionary = cv2.kmeans(desc, dictionary_size, None, criteria, 1, flags)
    np.save(params.BOW_DICT_PATH, dictionary)

    return dictionary

program_start = cv2.getTickCount()
# get training set
print("Loading images...")
start = cv2.getTickCount()

# must specify data path names in same order as class names
paths = [params.DATA_training_path_neg, params.DATA_training_path_pos]
# get class names
class_names = [get_class_name(class_number) for class_number in range(len(params.DATA_CLASS_NAMES))]
# again, must specify in same order as class names
sampling_sizes = [params.DATA_training_sample_count_neg, params.DATA_training_sample_count_pos]

# do we want to take samples only centric to the example image or ramdonly?
# No - for background -ve images (first class)
# Yes - for object samples +ve images (second class)
sample_from_centre = [False, True]

# perform image loading
imgs_data = load_images(paths, class_names, sampling_sizes, sample_from_centre,
                        params.DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES, params.DATA_WINDOW_SIZE)

print(("Loaded {} image(s)".format(len(imgs_data))))
print_duration(start)

# BOW feature construction
print("Computing descriptors...")
start = cv2.getTickCount()
[img_data.compute_bow_descriptors() for img_data in imgs_data]
print_duration(start)

print("Clustering...")
start = cv2.getTickCount()
dictionary = generate_dictionary(imgs_data, params.BOW_dictionary_size)
print_duration(start)

print("Generating histograms...")
start = cv2.getTickCount()
[img_data.generate_bow_hist(dictionary) for img_data in imgs_data]
print_duration(start)

# train an SVM based on these norm_features
print("Training SVM...")
start = cv2.getTickCount()

# define SVM parameters
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(params.BOW_SVM_kernel)

# compile samples for each training image
samples = get_bow_histograms(imgs_data)

# get class label for each training image
class_labels = get_class_labels(imgs_data)

# set termination criteria for SVM training
svm.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, params.BOW_SVM_max_training_iterations, 1.e-06))

# perform auto training for the SVM and save
svm.trainAuto(samples, cv2.ml.ROW_SAMPLE, class_labels, kFold = 10, balanced = True)
svm.save(params.BOW_SVM_PATH)

# measure performance of the SVM over some training data
output = svm.predict(samples)[1].ravel()
error = (np.absolute(class_labels.ravel() - output).sum()) / float(output.shape[0])

# we are succesful if our prediction > than random
if error < (1.0 / len(params.DATA_CLASS_NAMES)):
    print("Trained SVM obtained {}% training set error".format(round(error * 100,2)))
    print("-- meaining the SVM got {}% of the training examples correct!".format(round((1.0 - error) * 100,2)))
else:
    print("Failed to train SVM. {}% error".format(round(error * 100,2)))

print_duration(start)

print(("Finished training BOW detector. {}".format(format_time(get_elapsed_time(program_start)))))
