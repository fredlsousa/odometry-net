import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
import time

plot = True

if plot:
    import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

width = 320
height = 240

path = 'test_data'

histogram = True
scatter_plot = False
path_plot = False
boxplot = False

# Load pretrain mode
model = tf.keras.models.load_model('resnet-newdata-320.h5')

test_annot = pd.read_csv("test_annot.csv")
predictions_list = []
dirlist = test_annot['filename']
inference_time = []

with tqdm(total=len(dirlist)) as pbar:
    for file in dirlist:
        # Load image by OpenCV
        img = cv2.imread(path + '/' + file)

        # Resize to respect the input_shape
        inp = cv2.resize(img, (width, height))

        # Convert img to RGB
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

   	# Converting RGB image to Tensor
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)

        # Add dims to rgb_tensor
        rgb_tensor = tf.expand_dims(rgb_tensor, axis=0)

        # Predict and calculate time
        t = time.process_time()
        predictions = model.predict(rgb_tensor)
        elapsed_time = time.process_time() - t
        inference_time.append(elapsed_time)
        # print("Inference Time(s): ", elapsed_time)
        # print(predictions)
        predictions_list.append(predictions)
        pbar.update(1)

mean_inference_t = np.mean(inference_time)
print("Mean Inference Time(s): ", mean_inference_t)


x_gt = test_annot['x']
y_gt = test_annot['y']
# theta_gt = test_annot['theta']

i = 0
preds_dist = []
x_pred_list = []
y_pred_list = []

for prediction in predictions_list:
    x_pred = prediction[0][0]
    y_pred = prediction[0][1]
    x_pred_list.append(x_pred)
    y_pred_list.append(y_pred)
    theta_pred = prediction[0][2]
    dist = np.sqrt(((x_pred - x_gt[i]) ** 2) + ((y_pred - y_gt[i]) ** 2))
    # Debug
    # print("(%.2f,%.2f)|(%.2f,%.2f)"%(x_pred,y_pred,x_gt[i],y_gt[i]))
    # print("Dist = ",  dist)
    preds_dist.append(dist)
    i += 1

mean = np.mean(preds_dist)
std_deviation = np.std(preds_dist)

print("Mean: ", mean)
print("Standard Deviation: ", std_deviation)

# Plotting model test results with matplotlib
if plot:
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    gt_list = []

    j = 0
    if scatter_plot:
        for prediction in predictions_list:
            x_pred = prediction[0][0]
            y_pred = prediction[0][1]
            ax1.scatter(x_pred, y_pred, c='b')
            ax1.scatter(x_gt[j], y_gt[j], c='r')

            gt_list.append([x_gt[j], y_gt[j]])
            j += 1
    if boxplot:
        box = ax1.boxplot(preds_dist, notch=True, patch_artist=True)
        plt.savefig("boxplot-320-net.jpg")

    if histogram:
        ax1.hist(preds_dist, bins=10)
        plt.savefig("histogram-320-net.jpg")


if path_plot:
    plt.plot(x_gt, y_gt, color="red")
    plt.plot(x_pred_list, y_pred_list, color="blue")
    plt.savefig("paths_prediction_3.png")

