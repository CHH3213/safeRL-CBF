#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========safe-rl-cbf===============
@File: robotarium_get.py
@Time: 2022/3/10 上午9:19
@Author: chh3213
@Description: get robot state from gazebo
========Above the sun, full of fire!=============
"""
from __future__ import print_function, absolute_import, division
import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, GetModelState  # 设置模型状态、得到模型状态
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import *
from gazebo_msgs.srv import SetModelStateRequest
from geometry_msgs.msg import Pose, Twist
import time
import random
import numpy as np
from rotate_calculation import Rotate


class Robotarium:
    def __init__(self):
        # gazebo topic
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_cb)
        # gazebo服务
        self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # 指定服务名来调用服务
        self.set_model_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.Rot = Rotate()
        # self.rate = rospy.Rate(10)

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
        # self.rate.sleep()
        return [model_position, model_attitude, model_pose0, model_linear, model_angular]

    def resetWorld(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_world service call failed")

    def reset_agent_state(self, ddr_list):
        """
        :param ddr_list: 列表， 元素为car类
        :return:
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            vel = Twist()
            vel.linear.x = 0
            vel.linear.y = 0
            vel.linear.z = 0
            vel.angular.x = 0
            vel.angular.y = 0
            vel.angular.z = 0
            my_model = SetModelStateRequest()
            for i, ddr in enumerate(ddr_list):
                ddr.car_vel.publish(vel)
                my_model.model_state.model_name = ddr.name

                if i == 0:
                    x1, y1 = 3. * np.random.rand() - 1.5, 3. * np.random.rand() - 1.5

                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    # my_model.model_state.pose.position.z = 0.09

                if ddr.name == 'ddr_' + str(i) and i > 0:
                    x1, y1 = 2.6 * np.random.rand() - 1.0, 2.6 * np.random.rand() - 1.0

                    # x1, y1 = random.uniform(-2, 2), random.uniform(0, 3) # before 2021/11/22
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    # my_model.model_state.pose.position.z = 0.09


                else:
                    assert 'car‘s name may be properly wrong.'
                import tf
                th = random.uniform(-np.pi, np.pi)
                quart = tf.transformations.quaternion_from_euler(0, 0, th)
                # my_model.model_state.pose.position.z = 0.09
                my_model.model_state.pose.orientation.x = quart[0]
                my_model.model_state.pose.orientation.y = quart[1]
                my_model.model_state.pose.orientation.z = quart[2]
                my_model.model_state.pose.orientation.w = quart[3]
                my_model.model_state.twist.linear.x = 0.0
                my_model.model_state.twist.linear.y = 0.0
                my_model.model_state.twist.linear.z = 0.0
                my_model.model_state.twist.angular.x = 0.0
                my_model.model_state.twist.angular.y = 0.0
                my_model.model_state.twist.angular.z = 0.0
                my_model.model_state.reference_frame = "world"
                self.set_model_proxy(my_model)
                rospy.logdebug("Set model pose1 @ ({},{},{})".format(x1, y1, th))
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_modelState service call failed")

    def reset_target(self, target_pos):
        """
        :param target_pos: 需要设定的目标点位置（x,y)
        :return:
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            my_model = SetModelStateRequest()
            my_model.model_state.model_name = 'end_mark'
            x1, y1 = target_pos[0], target_pos[1]
            my_model.model_state.pose.position.x = x1
            my_model.model_state.pose.position.y = y1
            my_model.model_state.reference_frame = "world"
            self.set_model_proxy(my_model)
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_modelState service call failed")

    def reset_barrier(self, barrier_pos_list):
        """
        :param barrier_pos_list: 需要设定的障碍物位置（x,y)
        :return:
        """
        rospy.wait_for_service('/gazebo/set_model_state')
        for i, barrier in enumerate(barrier_pos_list):
            try:
                my_model = SetModelStateRequest()
                my_model.model_state.model_name = 'barrier_' + str(i)
                x1, y1 = barrier[0], barrier[1]
                my_model.model_state.pose.position.x = x1
                my_model.model_state.pose.position.y = y1
                my_model.model_state.reference_frame = "world"
                self.set_model_proxy(my_model)
            except rospy.ServiceException as e:
                rospy.logerr("/gazebo/reset_modelState service call failed")

    def resetSim(self):
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            self.reset_simulation()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/reset_simulation service call failed")

    def pausePhysics(self):
        rospy.wait_for_service("/gazebo/pause_physics")  # 等待服务器连接
        try:
            self.pause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

    def unpausePhysics(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

    def _model_states_cb(self, data):
        self.model_states = data
