#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========IEEE_TAI_DDR===============
@File: process--测试简单代码专用
@Time: 2021/11/14 上午10:54
@Author: chh3213
========Above the sun, full of fire!=============
"""
import  numpy as np
import numpy as np
import random

'''# 1.测试ones用法'''
# l = np.ones((3, 2)) * 4
# print(l)
'''# 2.测试列表乘系数'''
# done_n = [True] * 3
# print(done_n)
'''# 3.测试random'''
# print(np.random.random(2) * 8 - 4)
'''# 4.测试choice'''
# choose_target = np.array([[2, 2], [3, 3], [4, 4], [-2, 2], [-3, 3], [4, -4]])
# target = random.choice(choose_target)
# print(target)
# print(random.choice(range(5)))

'''# 5.测试寻找第二小的数的索引'''
# arr = np.random.rand(10)
# print(arr)
# # 第2大数值
# max2 = np.sort(arr)[-2]
# # 第2大索引
# max_index2 = np.argsort(arr)[-2]
# print(max_index2)
# x = np.array([3, 1, 2])
# index = np.argsort(x)
#
# print(x[index[0]])

"""6.测试list[None]"""
# list = np.array([21, 2, 3])
# print(list[None])
# a = np.array([[11, 12, 13, 14],
#               [21, 22, 23, 24],
#               [31, 32, 33, 34],
#               [41, 42, 43, 44],
#               ])

# print('0维为None:')
# print(a[None, 0:4])
# print('1维为None:')
# print(a[0:4, None])

'''.测试dict'''
walls = dict(wall_0=np.array([0, 3.1]), wall_1=np.array([0, -3.1]), wall_2=np.array([-3.1, 0]), wall_3=np.array([3.1, 0]))
print(walls['wall_1'])