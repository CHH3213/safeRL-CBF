import scipy.io as sio
import numpy as np
import sys
sys.path.append('..')
#cbf
data = sio.loadmat(
    '/home/YHS/ros_ws/src/IEEE_TAI_DDR/ddr_control/scripts/algos/save/td3_add_obs_2021-11-30-18:26:30/data/td3_nocbf_nod_test_01-16:15:04.mat')
#nocbf
# data = sio.loadmat(
#     '/home/YHS/ros_ws/src/IEEE_TAI_DDR/ddr_control/scripts/algos/save/td3_add_obs_2021-11-30-18:26:30/data/td3_nocbf_nod_test_01-16:47:11.mat')
print(data.keys())
print(data['count_wall'])
print(data['count_agent'])
