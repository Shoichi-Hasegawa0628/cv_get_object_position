#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard Library
import yaml
import cv2
import time
import numpy as np
import statistics
import math

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
# from yolo_ros_msgs.msg import BoundingBoxes, BoundingBox


class CvGetObjectPosition():
    def __init__(self, yolo_bb_topic="/output/bboxes",
                 point_cloud_topic="/hsrb/head_rgbd_sensor/depth_registered/rectified_points",
                 marker_array_topic="/view_target_objects_points"):

        with open('../../config/target_objects.yaml', 'r') as yml:
            self.target_objects = yaml.load(yml)['names']

        self.iter = 3
        self.neighborhood_value = 24
        self.flag = 0
        self.sum = []
        self.x = []
        self.y = []
        self.z = []
        self.id_count = 0
        self.target_objects_positions = {}
        self.target_obejcts_positions_tmp = []
        for i in range(len(self.target_objects)):
            self.target_obejcts_positions_tmp.append([[], [], []])
        self.bounding_topic_name = yolo_bb_topic
        # self.bounding_topic_name = "/yolov5_ros/output/bounding_boxes"
        self.point_cloud_name = point_cloud_topic
        self.marker_array_data = MarkerArray()
        self.pub_marker_array = rospy.Publisher(marker_array_topic, MarkerArray, queue_size=1)
        self.tflistener = TransformListener()

    def main(self):
        self.setup_subscriber()

        while self.flag == 0:
            pass
            # rospy.loginfo("waiting.")
            # if self.flag == 1:
                # rospy.loginfo("Calculatation is OK.")
                # rospy.loginfo("Use this positions:{}".format(self.target_objects_positions))
                # self.subscriber_for_point_cloud.unregister()
                # self.subscriber_for_bounding_box.unregister()
                # self.visualization_target_object_positions()
                # return
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
        if self.is_target_class_in_bboxes(message.boxes):
            for box in message.boxes:
                if not box.name in self.target_objects:
                    continue

                cx = (int(box.min_x) + (int(box.max_x) - int(box.min_x)) / 2)
                cy = (int(box.min_y) + (int(box.max_y) - int(box.min_y)) / 2)

                pose_stamped = PoseStamped()

                position = self.get_point(int(cx), int(cy))
                if position is False:
                    continue

                # rospy.loginfo("cX:{} cY:{} cZ:{}".format(position[0], position[1], position[2]))  # Camera Coordinate

                if not (position is None):
                    tf_buffer = tf2_ros.Buffer()
                    tf_listener = tf2_ros.TransformListener(tf_buffer)
                    try:
                        self.trans = tf_buffer.lookup_transform('map', 'head_rgbd_sensor_rgb_frame', rospy.Time(0),
                                                                rospy.Duration(1.0))
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
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

                    # 物体ごとに格納場所を分ける作業
                    idx = self.target_objects.index(box.name)
                    self.target_obejcts_positions_tmp[idx][0].append(median_neighborhood_points_x)
                    self.target_obejcts_positions_tmp[idx][1].append(median_neighborhood_points_y)
                    self.target_obejcts_positions_tmp[idx][2].append(median_neighborhood_points_z)

                    # 格納した物体の座標のみに適用
                    median_x = statistics.median(self.target_obejcts_positions_tmp[idx][0])
                    median_y = statistics.median(self.target_obejcts_positions_tmp[idx][1])
                    median_z = statistics.median(self.target_obejcts_positions_tmp[idx][2])

                    # rospy.loginfo(
                    #     "Class:{} medX:{:.3f} medY:{:.3f} medZ:{:.3f}".format(box.Class, median_x, median_y, median_z))
                    print(len(self.target_obejcts_positions_tmp[idx][0]))
                    # print(self.iter)


                    # もしいま処理している物体のiter回数が設定値と同じなら
                    if len(self.target_obejcts_positions_tmp[idx][0]) == self.iter:
                        self.target_objects_positions[box.name] = [median_x, median_y, median_z]
                        print(self.target_objects_positions)
                        # print(list(self.target_objects_positions.keys())[0])

                    # もし対象物全てに対して処理を終えているのなら
                    if len(list(self.target_objects_positions.keys())) == len(self.target_objects):
                        print("ALL FINISHED !")
                        rospy.loginfo("Calculatation is OK.")
                        rospy.loginfo("Use this positions:{}".format(self.target_objects_positions))
                        self.subscriber_for_point_cloud.unregister()
                        self.subscriber_for_bounding_box.unregister()
                        self.flag = 1


                    return

        # if self.is_target_class_in_bboxes(message.bounding_boxes):
        #     for box in message.bounding_boxes:
        #         if not box.Class in self.target_objects:
        #             continue
        #
        #         cx = (int(box.xmin) + (int(box.xmax) - int(box.xmin)) / 2)
        #         cy = (int(box.ymin) + (int(box.ymax) - int(box.ymin)) / 2)
        #
        #         pose_stamped = PoseStamped()
        #
        #         position = self.get_point(int(cx), int(cy))
        #         if position is False:
        #             continue
        #
        #         # rospy.loginfo("cX:{} cY:{} cZ:{}".format(position[0], position[1], position[2]))  # Camera Coordinate
        #
        #         if not (position is None):
        #             tf_buffer = tf2_ros.Buffer()
        #             tf_listener = tf2_ros.TransformListener(tf_buffer)
        #             try:
        #                 self.trans = tf_buffer.lookup_transform('map', 'head_rgbd_sensor_rgb_frame', rospy.Time(0),
        #                                                         rospy.Duration(1.0))
        #             except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #                 return
        #
        #             neighborhood_points = self.get_neighborhood_points(cx, cy, self.neighborhood_value)
        #             if neighborhood_points is False:
        #                 continue
        #
        #             for i in range(len(neighborhood_points)):
        #                 neighborhood_points[i].append(position[i])
        #
        #             transformed_points = [[], [], []]
        #             for i in range(len(neighborhood_points[0])):
        #                 pose_stamped.header = self._point_cloud_header
        #                 pose_stamped.pose.position.x = neighborhood_points[0][i]
        #                 pose_stamped.pose.position.y = neighborhood_points[1][i]
        #                 pose_stamped.pose.position.z = neighborhood_points[2][i]
        #                 # rospy.loginfo("cX:{:.3f} cY:{:.3f} cZ:{:.3f}".format(_pose_stamped.pose.position.x,
        #                 #                                                   _pose_stamped.pose.position.y,
        #                 #                                                   _pose_stamped.pose.position.z)) # Possible for converting coordinate (Camera Coordinate)
        #
        #                 transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, self.trans)
        #
        #                 # rospy.loginfo("mX:{:.3f} mY:{:.3f} mZ:{:.3f}".format(transformed.pose.position.x,
        #                 #                                                      transformed.pose.position.y,
        #                 #                                                      transformed.pose.position.z))  # Map Coordinate
        #
        #                 transformed_points[0].append(transformed.pose.position.x)
        #                 transformed_points[1].append(transformed.pose.position.y)
        #                 transformed_points[2].append(transformed.pose.position.z)
        #
        #             # rospy.loginfo("Label: {}".format(box.Class))
        #
        #             for i in range(len(transformed_points[0])):
        #                 if not np.isnan(transformed_points[0][i]):
        #                     pass
        #                 else:
        #                     return
        #
        #             median_neighborhood_points_x = statistics.median(transformed_points[0])
        #             median_neighborhood_points_y = statistics.median(transformed_points[1])
        #             median_neighborhood_points_z = statistics.median(transformed_points[2])
        #
        #             # 物体ごとに格納場所を分ける作業
        #             idx = self.target_objects.index(box.Class)
        #             self.target_obejcts_positions_tmp[idx][0].append(median_neighborhood_points_x)
        #             self.target_obejcts_positions_tmp[idx][1].append(median_neighborhood_points_y)
        #             self.target_obejcts_positions_tmp[idx][2].append(median_neighborhood_points_z)
        #
        #             # 格納した物体の座標のみに適用
        #             median_x = statistics.median(self.target_obejcts_positions_tmp[idx][0])
        #             median_y = statistics.median(self.target_obejcts_positions_tmp[idx][1])
        #             median_z = statistics.median(self.target_obejcts_positions_tmp[idx][2])
        #
        #             # rospy.loginfo(
        #             #     "Class:{} medX:{:.3f} medY:{:.3f} medZ:{:.3f}".format(box.Class, median_x, median_y, median_z))
        #             print(len(self.target_obejcts_positions_tmp[idx][0]))
        #             # print(self.iter)
        #
        #
        #             # もしいま処理している物体のiter回数が設定値と同じなら
        #             if len(self.target_obejcts_positions_tmp[idx][0]) == self.iter:
        #                 self.target_objects_positions[box.Class] = [median_x, median_y, median_z]
        #                 print(self.target_objects_positions)
        #                 # print(list(self.target_objects_positions.keys())[0])
        #
        #             # もし対象物全てに対して処理を終えているのなら
        #             if len(list(self.target_objects_positions.keys())) == len(self.target_objects):
        #                 print("ALL FINISHED !")
        #                 rospy.loginfo("Calculatation is OK.")
        #                 rospy.loginfo("Use this positions:{}".format(self.target_objects_positions))
        #                 self.subscriber_for_point_cloud.unregister()
        #                 self.subscriber_for_bounding_box.unregister()
        #                 self.flag = 1
        #
        #
        #             return


    def point_cloud_callback(self, point_cloud):
        self._point_cloud = pointcloud2_to_xyz_array(point_cloud, False)
        self._point_cloud_header = point_cloud.header

    def get_point(self, x, y):
        try:
            return self._point_cloud[y][x]
        except:
            rospy.loginfo("GET POINT ERROR")
            return False

    def is_target_class_in_bboxes(self, bboxes):
        for bbox in bboxes:
            if bbox.name in self.target_objects:
                if bbox.name in list(self.target_objects_positions.keys()):
                    return False
                else:
                    return True
        return False

    # def is_target_class_in_bboxes(self, bboxes):
    #     for bbox in bboxes:
    #         if bbox.Class in self.target_objects:
    #             if bbox.Class in list(self.target_objects_positions.keys()):
    #                 return False
    #             else:
    #                 return True
    #     return False

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
        # print(neighborhood_points[0])
        return neighborhood_points

    def visualization_target_object_positions(self):
        for o in range(len(self.target_objects)):
            # print("{}".format(o))
            # text_marker, pose_marker = self.init_marker()
            pose_marker = self.init_marker()

            # text_marker.text = self.target_objects[o]
            # text_marker.pose.position.x = self.target_objects_positions[o][0]
            # text_marker.pose.position.y = self.target_objects_positions[o][1]
            # text_marker.pose.position.z = self.target_objects_positions[o][2] + 0.5
            # text_marker.color.r = 0.0
            # text_marker.color.g = 0.8
            # text_marker.color.b = 0.0

            pose_marker.pose.position.x = self.target_objects_positions[self.target_objects[o]][0]
            pose_marker.pose.position.y = self.target_objects_positions[self.target_objects[o]][1]
            pose_marker.pose.position.z = self.target_objects_positions[self.target_objects[o]][2]
            pose_marker.color.r = 0.8
            pose_marker.color.g = 0.0
            pose_marker.color.b = 0.0

            # self.id_count += 1
            # text_marker.id = self.id_count
            # text_marker.ns = "marker" + str(self.id_count)
            # self.marker_array_data.markers.append(text_marker)
            self.id_count += 1
            pose_marker.id = self.id_count
            pose_marker.ns = "marker" + str(self.id_count)
            self.marker_array_data.markers.append(pose_marker)
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.pub_marker_array.publish(self.marker_array_data)
            r.sleep()

    def init_marker(self):
        # def_text_marker = Marker()
        # def_text_marker.type = Marker.TEXT_VIEW_FACING
        # def_text_marker.header.frame_id = "map"
        # def_text_marker.header.stamp = rospy.get_rostime()
        # def_text_marker.action = Marker.ADD
        # def_text_marker.scale.x = 0.40
        # def_text_marker.scale.y = 0.40
        # def_text_marker.scale.z = 0.40
        # def_text_marker.lifetime = rospy.Duration(100)
        # def_text_marker.color.a = 1

        def_pose_marker = Marker()
        # def_pose_marker.type = Marker.SPHERE
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
        def_pose_marker.pose.orientation.w = 0.0

        # return def_text_marker, def_pose_marker
        return def_pose_marker


if __name__ == "__main__":
    rospy.init_node('cv_get_object_position')
    cv_get_object_position = CvGetObjectPosition()
    cv_get_object_position.main()
    rospy.spin()
