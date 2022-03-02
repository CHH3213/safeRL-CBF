# -*- coding: utf-8 -*-
"""
2021/11/22
@modified: chh
"""
import matplotlib.pyplot as plt
import numpy as np
lines1 = np.loadtxt('./td3_qp_reward_st_2021-11-29-00:40:47.txt',
                    comments="#", delimiter="\n", unpack=False)
lines2 = np.loadtxt('./td3_qp_reward_st_ddr0_2021-11-29-00:40:47.txt',
                    comments="#", delimiter="\n", unpack=False)
lines3 = np.loadtxt('./td3_qp_reward_st_ddr1_2021-11-29-00:40:47.txt',
                    comments="#", delimiter="\n", unpack=False)
plt.figure()
plt.subplot(311)
plt.plot(lines1)
plt.subplot(312)
plt.plot(lines2)
plt.subplot(313)
plt.plot(lines3)
plt.show()
