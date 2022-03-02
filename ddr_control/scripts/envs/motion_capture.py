# -*-coding:utf-8-*-
'''
通过gazebo来reset状态
'''
from __future__ import print_function, absolute_import, division
import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Twist
import time
import random
import numpy as np
from envs.rotate_calculation import Rotate

# FIXME:motion capture
class Motion_capture:
    def __init__(self):
        # gazebo topic
        Motion_capture_name = ''
        rospy.Subscriber(Motion_capture_name, ModelStates, self._model_states_cb)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.Rot = Rotate()

    def setModelState(self, model_state):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.set_model_state_proxy(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def get_state(self, model_name):
        model = self.model_states.name.index(model_name)  #

        model_pose = self.model_states.pose[model]
        model_twist = self.model_states.twist[model]
        model_position = [model_pose.position.x, model_pose.position.y, model_pose.position.z]
        # 四元数
        model_pose0 = [model_pose.orientation.x, model_pose.orientation.y, model_pose.orientation.z,
                       model_pose.orientation.w]
        roll, pitch, yaw = self.Rot.quaternion_to_euler(model_pose.orientation.x, model_pose.orientation.y,
                                                        model_pose.orientation.z, model_pose.orientation.w)
        # 欧拉角
        model_attitude = [roll, pitch, yaw]
        model_linear = [model_twist.linear.x, model_twist.linear.y, model_twist.linear.z]
        model_angular = [model_twist.angular.x, model_twist.angular.y, model_twist.angular.z]
        # print([model_position,model_orientation,model_linear,model_angular])
        # 位置，姿态，线速度，角速度
        return [model_position, model_attitude, model_pose0, model_linear, model_angular]



    def _model_states_cb(self, data):
        self.model_states = data
