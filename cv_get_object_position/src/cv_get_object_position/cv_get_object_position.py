#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard Library
import yaml
import cv2
import time
import numpy as np
import statistics
import math
import csv

# ROS Library
import rospy
import tf
from tf import TransformListener
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array

# Third Package
from yolov5_ros_msgs.msg import BoundingBoxes, BoundingBox


class CvGetObjectPosition():
    def __init__(self, yolo_bb_topic="/output/bboxes",
                 point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
                 marker_array_topic="/view_target_objects_points"):

        with open('../../config/target_objects.yaml', 'r') as yml:
            object_yaml = yaml.load(yml, Loader=yaml.SafeLoader)
            self.target_objects = object_yaml['names']
            self.target_objects_number = object_yaml['number']

        self.neighborhood_value = 24
        self.flag = 0
        self.id_count = 0
        self.bounding_topic_name = yolo_bb_topic
        self.point_cloud_name = point_cloud_topic
        self.marker_array_data = MarkerArray()
        self.pub_marker_array = rospy.Publisher(marker_array_topic, MarkerArray, queue_size=1)
        # self.tflistener = TransformListener()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def main(self):
        self.setup_subscriber()

        while self.flag == 0:
            pass

        if self.flag == 1:
            self.visualization_target_object_positions()

    def setup_subscriber(self):
        self.subscriber_for_point_cloud = rospy.Subscriber(
            self.point_cloud_name,
            PointCloud2,
            self.point_cloud_callback,
            queue_size=1
        )

        self.subscriber_for_bounding_box = rospy.Subscriber(
            self.bounding_topic_name,
            BoundingBoxes,
            self.bounding_callback,
            queue_size=1
        )
        return

    def bounding_callback(self, message):
        target_objects_positions = {}
        # ダブリ判定
        # if self.is_target_class_in_bboxes(message.boxes):

        # 1枚の画像に対象物のラベルが全て含まれているか
        if self.is_all_target_in_image(message.boxes):
            for j, box in enumerate(message.boxes):
                if not box.name in self.target_objects:
                    continue

                cx = (int(box.min_x) + (int(box.max_x) - int(box.min_x)) / 2)
                cy = (int(box.min_y) + (int(box.max_y) - int(box.min_y)) / 2)

                pose_stamped = PoseStamped()

                position = self.get_point(int(cx), int(cy))
                if position is False:
                    continue

                print(position)
                # rospy.loginfo("cX:{} cY:{} cZ:{}".format(position[0], position[1], position[2]))  # Camera Coordinate

                if not (position is None):
                    # tf_buffer = tf2_ros.Buffer()
                    # tf_listener = tf2_ros.TransformListener(tf_buffer)
                    try:
                        self.trans = self.tf_buffer.lookup_transform('map', 'head_rgbd_sensor_rgb_frame', rospy.Time(0),
                                                                rospy.Duration(1.0))
                    except tf.LookupException as e_1:
                        print(e_1)
                        return

                    except tf.ConnectivityException as e_2:
                        print(e_2)
                        return

                    except tf.ExtrapolationException as e_3:
                        print(e_3)
                        return

                    neighborhood_points = self.get_neighborhood_points(cx, cy, self.neighborhood_value)
                    if neighborhood_points is False:
                        continue

                    for i in range(len(neighborhood_points)):
                        neighborhood_points[i].append(position[i])

                    transformed_points = [[], [], []]
                    for i in range(len(neighborhood_points[0])):
                        pose_stamped.header = self._point_cloud_header
                        pose_stamped.pose.position.x = neighborhood_points[0][i]
                        pose_stamped.pose.position.y = neighborhood_points[1][i]
                        pose_stamped.pose.position.z = neighborhood_points[2][i]
                        # rospy.loginfo("cX:{:.3f} cY:{:.3f} cZ:{:.3f}".format(_pose_stamped.pose.position.x,
                        #                                                   _pose_stamped.pose.position.y,
                        #                                                   _pose_stamped.pose.position.z)) # Possible for converting coordinate (Camera Coordinate)

                        transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.trans)

                        # rospy.loginfo("mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x,
                        #                                                      transformed.pose.position.y,
                        #                                                      transformed.pose.position.z))  # Map Coordinate

                        transformed_points[0].append(transformed.pose.position.x)
                        transformed_points[1].append(transformed.pose.position.y)
                        transformed_points[2].append(transformed.pose.position.z)

                    # rospy.loginfo("Label: {}".format(box.Class))

                    for i in range(len(transformed_points[0])):
                        if not np.isnan(transformed_points[0][i]):
                            pass
                        else:
                            return

                    median_neighborhood_points_x = statistics.median(transformed_points[0])
                    median_neighborhood_points_y = statistics.median(transformed_points[1])
                    median_neighborhood_points_z = statistics.median(transformed_points[2])

                    target_objects_positions[j] = [box.name, median_neighborhood_points_x, median_neighborhood_points_y, median_neighborhood_points_z]
                    print(target_objects_positions)

            # 取得した物体ラベルを格納
            value_list = list(target_objects_positions.values())
            actual_objects = []
            for i in range(len(value_list)):
                actual_objects.append(value_list[i][0])

            # 対象の物体が入っているか
            for i in range(len(self.target_objects)):
                if not self.target_objects[i] in actual_objects:
                    return
    
            print("Correct object name")
            print("\n")

            # 指定した個数入っているか
            for i in range(len(actual_objects)):
                object_idex = self.target_objects.index(actual_objects[i])
                print(actual_objects)
                print(self.target_objects_number[object_idex])
                print(actual_objects.count(self.target_objects[object_idex]))
                if self.target_objects_number[object_idex] != actual_objects.count(self.target_objects[object_idex]):
                    print("Fail number")
                    return

            print("Correct object number")

            print("ALL FINISHED !")
            rospy.loginfo("Calculatation is OK.")
            rospy.loginfo("Use this positions:{}".format(target_objects_positions))
            self.target_objects_positions = target_objects_positions
            self.save_data()
            self.subscriber_for_point_cloud.unregister()
            self.subscriber_for_bounding_box.unregister()
            self.flag = 1

            return

    def point_cloud_callback(self, point_cloud):
        self._point_cloud = pointcloud2_to_xyz_array(point_cloud, False)
        self._point_cloud_header = point_cloud.header

    def get_point(self, x, y):
        try:
            return self._point_cloud[y][x]
        except:
            rospy.loginfo("GET POINT ERROR")
            return False

    def is_all_target_in_image(self, bboxes):
        # bboxesに含まれるラベルをlist化
        bbox_labels = []
        for bbox in bboxes:
            bbox_labels.append(bbox.name)

        # TargetのListの物体が検出したBBの中に全て含まれるか
        for i in range(len(self.target_objects)):
            if self.target_objects[i] in bbox_labels:
                pass
            else:
                return False

        return True

    def get_neighborhood_points(self, cx, cy, value):
        """
        画像座標系
         u →
        v
        ↓
        """
        neighborhood_points = [[], [], []]
        width = int(math.sqrt(value + 1))
        for i in range(value + 1):
            for j in range(width):  # v
                for k in range(width):  # u
                    if j + 1 == k + 1 == math.ceil(width + 1):  # 中心点の判定
                        continue
                    else:

                        # uの取得
                        u = cx - (math.floor(width / 2)) + k

                        # vの取得
                        v = cy - (math.floor(width / 2)) + j

                        # カメラ座標の値を取得
                        position = self.get_point(int(u), int(v)).tolist()
                        if position is False:
                            return False

                        for l in range(len(neighborhood_points)):
                            neighborhood_points[l].append(position[l])
        return neighborhood_points

    def visualization_target_object_positions(self):
        boxes = list(self.target_objects_positions.keys())
        value_list = list(self.target_objects_positions.values())
        for i in range(len(boxes)):
            pose_marker = self.init_marker()
            pose_marker.pose.position.x = value_list[i][2]
            pose_marker.pose.position.y = value_list[i][3]
            pose_marker.pose.position.z = value_list[i][4]

            pose_marker.color.r = 0.8
            pose_marker.color.g = 0.0
            pose_marker.color.b = 0.0

            self.id_count += 1
            pose_marker.id = self.id_count
            pose_marker.ns = "marker" + str(self.id_count)
            self.marker_array_data.markers.append(pose_marker)

        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.pub_marker_array.publish(self.marker_array_data)
            r.sleep()
        

        

    def init_marker(self):

        def_pose_marker = Marker()
        def_pose_marker.type = Marker.CYLINDER
        def_pose_marker.header.frame_id = "map"
        def_pose_marker.header.stamp = rospy.get_rostime()
        def_pose_marker.action = Marker.ADD
        def_pose_marker.scale.x = 0.1
        def_pose_marker.scale.y = 0.1
        def_pose_marker.scale.z = 0.1
        def_pose_marker.lifetime = rospy.Duration(100)
        def_pose_marker.color.a = 1
        def_pose_marker.id = self.id_count
        def_pose_marker.ns = "marker"

        def_pose_marker.pose.orientation.x = 0.0
        def_pose_marker.pose.orientation.y = 0.0
        def_pose_marker.pose.orientation.z = 0.0
        def_pose_marker.pose.orientation.w = 1.0

        return def_pose_marker

    def save_data(self):
        data = []
        labels = list(self.target_objects_positions.keys())
        positions = list(self.target_objects_positions.values())
        with open('../../data/object_position.csv', 'w') as f:
            writer = csv.writer(f)
            for i in range(len(labels)):
                positions[i].insert(0, labels[i])
                writer.writerow(positions[i])

if __name__ == "__main__":
    rospy.init_node('cv_get_object_position')
    cv_get_object_position = CvGetObjectPosition()
    cv_get_object_position.main()
    # rospy.spin()
