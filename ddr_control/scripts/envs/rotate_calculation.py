# -*-coding:utf-8-*-
'''
#############chh###########
旋转前后的关系转换
#########################
'''
import numpy as np
import math
from math import *
class Rotate:
    def __init__(self):
        pass
    def quaternion_to_euler(self,x, y, z, w):
        '''
        四元数转化为欧拉角:与tf库默认方向一致
        返回欧拉角数组
        '''

        X = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        Y = math.asin(2 * (w * y - x * z))
        Z = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
        # 使用 tf 库
        import tf
        (X, Y, Z) = tf.transformations.euler_from_quaternion([x, y, z, w])
        return np.array([X, Y, Z])

    def euler_to_quaternion(self,roll, pitch, yaw):
        '''
        欧拉角转化为四元数
        返回四元数数组
        '''
        x=sin(pitch/2)*sin(yaw/2)*cos(roll/2)+cos(pitch/2)*cos(yaw/2)*sin(roll/2)
        y=sin(pitch/2)*cos(yaw/2)*cos(roll/2)+cos(pitch/2)*sin(yaw/2)*sin(roll/2)
        z=cos(pitch/2)*sin(yaw/2)*cos(roll/2)-sin(pitch/2)*cos(yaw/2)*sin(roll/2)
        w=cos(pitch/2)*cos(yaw/2)*cos(roll/2)-sin(pitch/2)*sin(yaw/2)*sin(roll/2)
        import tf
        (x, y, z, w) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        return np.array([x, y, z, w])

    # def quaternion_to_rotation_matrix(quat):
    #     '''
    #     四元数转换为旋转矩阵
    #     '''
    #     q = quat.copy()
    #     n = np.dot(q, q)
    #     if n < np.finfo(q.dtype).eps:
    #         return np.identity(4)
    #     q = q * np.sqrt(2.0 / n)
    #     q = np.outer(q, q)
    #     rot_matrix = np.array(
    #         [
    #         [1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
    #         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
    #         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
    #         [0.0, 0.0, 0.0, 1.0]
    #         ],
    #         dtype=q.dtype)
    #     return rot_matrix

    def quaternion_to_rotation_matrix(self,q):  # x, y ,z ,w
        '''
        四元数转换为旋转矩阵
        '''
        rot_matrix = np.array(
            [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2]),0],
             [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0]),0],
             [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1]),0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=q.dtype)
        return rot_matrix

    def eulerAnglesToRotationMatrix(self, theta):

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_x, np.dot(R_y, R_z))
        return R

    def quatProduct(self,q1, q2):
        '''
        四元数q1*q2
        q = q1*q2
        returns:旋转后的向量--四元数形式
        '''
        # 实部
        r1 = q1[3]
        r2 = q2[3]
        v1 = np.array([q1[0], q1[1], q1[2]])
        v2 = np.array([q2[0], q2[1], q2[2]])

        r = r1 * r2 - np.dot(v1, v2)
        v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
        q = np.array([v[0], v[1], v[2],r])

        return q

    def quatInvert(self,q):
        '''
        求四元数的逆
        '''
        q_star = np.array([-q[0],-q[1],-q[2],q[3]])
        q_modSquare = np.sum(np.square(q))
        q_invert = q_star/q_modSquare
        return q_invert

    def rotated_vector(self,q,v):
        '''
        input:q:四元数，v:旋转前向量-->[v,0]补齐为四元数
        求旋转后的向量，使用四元数
        '''
        v_rotated = self.quatProduct(self.quatProduct(q,v),self.quatInvert(q))
        return v_rotated


if __name__ == '__main__':
    rotate = Rotate()
    quat =rotate.euler_to_quaternion(np.pi/2,0,0)
    print(quat)
    v = np.array([0. ,   0.289, 0.05,  0. ])
    # 旋转后的向量--四元数
    v_r = rotate.rotated_vector(quat,v)
    # 旋转后的向量--通过旋转矩阵求解
    v_r2 = np.dot(rotate.quaternion_to_rotation_matrix(quat),v)
    print(v_r)
    print(v_r2)
#    print(pow(300,1/1.5))
