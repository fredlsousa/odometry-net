import cv2
import tensorflow as tf
import time
import numpy as np
import socket


# Comment the thre following lines in case GPU is not available for frame processing. It is also recommended to disable fast_fps variable in this case
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":

    fast_fps = False

    if fast_fps:    # if true, camera framerate is higher but if loop gets stuck in the prediciton, there'll be prediciton delays
        cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 num-buffers=300 ! '
                                    'video/x-raw,width=640,height=480,framerate=30/1 ! videorate ! '
                                    'video/x-raw,framerate=30/1 ! videoscale ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
    
    else:   # set fast_fps to false in slower prediciton systems ie. Raspberry Pi
        cap_send = cv2.VideoCapture(0)
    
    if not cap_send.isOpened():
        print('VideoCapture not opened')
        exit(0)

    model = tf.keras.models.load_model('resnet-newdata-320.h5')

    net_width = 320
    net_height = 240

    inference_time = []

    # i = 0

    try:
        while True:
            ret, img = cap_send.read()
            # img_name = "/workspace/ros_camera_info/imgs_4new/" + str(i) + ".jpg"
            # img = cv2.imread(img_name)

            if not ret:
                print('Empty frame! Exiting...')
                break

            inp = cv2.resize(img, (net_width, net_height))
            rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
            rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
            rgb_tensor = tf.expand_dims(rgb_tensor, axis=0)
            t = time.process_time()
            prediction = model.predict(rgb_tensor)
            # print("Prediction: ", prediction)
            elapsed_time = time.process_time() - t
            inference_time.append(elapsed_time)

            x_pred = prediction[0][0]
            y_pred = prediction[0][1]
            yawn_pred = prediction[0][2]
            # i += 1

    except KeyboardInterrupt:
        mean_inference_t = np.mean(inference_time)
        print("Mean Inference Time(s): ", mean_inference_t)
