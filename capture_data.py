import cv2
import math
import numpy as np
import rospy
from nav_msgs.msg import Odometry


class OdomSub(object):

    def __init__(self):
        self.topic_name = "odom"
        self.topic_data = None
        self.pose_w_cov = None
        self.only_pose = None
        self.position_covariance = None
        self.header = None
        self.twist_data = None
        self.twist_data_covariance = None
        self.orientation = None
        self.subs = rospy.Subscriber(self.topic_name, Odometry, self.odom_callback)
        rospy.loginfo("Subscribed to /" + str(self.topic_name))

    def odom_callback(self, msg):
        self.topic_data = msg
        self.pose_w_cov = msg.pose
        self.only_pose = msg.pose.pose.position
        self.position_covariance = msg.pose.covariance
        self.twist_data = msg.twist.twist
        self.twist_data_covariance = msg.twist.covariance
        self.orientation = msg.pose.pose.orientation

    def get_only_position(self):
        return self.only_pose

    def get_pose(self):
        return self.pose_w_cov

    def get_orientation(self):
        return self.orientation

    def get_position_covariance(self):
        return self.position_covariance

    def get_twist(self):
        return self.twist_data


def quaternion2euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


rospy.init_node('dataset_capturer', anonymous=True)
rate = rospy.Rate(100)

if __name__ == '__main__':

    odom_sub = OdomSub()
    rospy.sleep(1)
    cap = cv2.VideoCapture(0)
    annot_file = open("annotation.csv", "w")
    annot_file.write("filename,x,y,theta\n")

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    rospy.loginfo("Capturing Odom&Cam data...")

    while not rospy.is_shutdown():

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        odom_pose = odom_sub.get_only_position()
        odom_orientation = odom_sub.get_orientation()
        roll_x, pitch_y, yaw_z = quaternion2euler(odom_orientation.x, odom_orientation.y,
                                                  odom_orientation.z, odom_orientation.w)

        img_name = rospy.get_rostime()
        img_name = "imgs/" + str(img_name) + ".jpg"
        cv2.imwrite(img_name, frame)
        annot_file.write(img_name + "," + str(odom_pose.x) + "," + str(odom_pose.y) + "," + str(yaw_z) + "\n")

        rospy.sleep(0.5)

    cap.release()
