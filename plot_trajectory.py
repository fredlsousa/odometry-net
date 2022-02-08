import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Normalizing ORB_SLAM data to rf2o and resnet predicitons scale
def normalize_data(data_list, bound):
    data_list = np.array(data_list)
    min_val = np.min(data_list)
    max_val = np.max(data_list)
    norm_list = []
    for vl in data_list:
        norm_val = ((vl) / (max_val - min_val)) * bound
        norm_list.append(norm_val)

    return norm_list


path = pd.read_csv("path4new.csv")
gt_rf20 = pd.read_csv("file_gt.csv")
orb_slam = pd.read_csv("pose_file_orb_slam.csv")

x_path = path['x']
y_path = path['y']

x_rf20 = gt_rf20['x']
y_rf20 = gt_rf20['y']

x_orb = orb_slam['x']
y_orb = orb_slam['y']


norm_x_orb = normalize_data(x_orb, 3.5)
norm_y_orb = normalize_data(y_orb, 0.5)


filtered_rf2o = savgol_filter(y_rf20, len(y_rf20)-1, 3)
filtered_path = savgol_filter(y_path, len(y_path), 3)
filtered_orb = savgol_filter(norm_y_orb, len(norm_y_orb)-1, 3)

plt.rc('font', size=18)

distances_net = []
distances_orb = []

for i in range(0, len(vals)):

    x_p = x_path[i]
    y_p = filtered_path[i]

    x_r = x_rf20[i]
    y_r = filtered_rf2o[i]

    dist_net = np.sqrt(((x_p - x_r) ** 2) + ((y_p - y_r) ** 2))
    distances_net.append(dist_net)

for i in range(0, len(x_rf20)):
    x_r = x_rf20[i]
    y_r = filtered_rf2o[i]

    x_o = norm_x_orb[i]
    y_o = filtered_orb[i]

    dist_orb = np.sqrt(((x_o - x_r) ** 2) + ((y_o - y_r) ** 2))
    distances_orb.append(dist_orb)
    

mean_dist = np.mean(distances_net)
means_dist_orb = np.mean(distances_orb)
print("Mean Dist (net): ", mean_dist)
print("Mean Dist (orb): ", means_dist_orb)


red_patch = mpatches.Patch(color='red', label='ORB_SLAM2')
black_patch = mpatches.Patch(color='black', label='RF2O - GT')
blue_patch = mpatches.Patch(color='blue', label='ResNet50 - 320x240')
plt.legend(handles=[red_patch, black_patch, blue_patch])

plt.ylabel("[m]")
plt.xlabel("[m]")

plt.plot(x_path, filtered_path, color='blue', linewidth=4,)
plt.plot(x_rf20, filtered_rf2o, color='black', linewidth=4,)
plt.plot(norm_x_orb, filtered_orb, color='red', linewidth=4,)
plt.show()
