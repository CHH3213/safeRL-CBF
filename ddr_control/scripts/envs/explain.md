> ## 对于logger小车而言：
>

1. 角速度为负,逆时针转动，为正,顺时针转动
2. 姿态角向左为正，向右为负

## envs文件说明

### 1. gazebo_reset.py

> 用于模型的状态设置，初始化目标点等。

### 2.  rotate_calculation.py

> 姿态旋转相关函数


### 5. train_safe_env.py

> 训练所用环境，状态量为17维，加入了最近的obstacle和wall的坐标,配套`test_safe_env.py`

### 6. train_env_multi.py

> 多个TD3联合训练，与`test_env_multi.py`配套


### 8.  train_safe_env_backup_11-30.py

> 训练所用环境，状态量维13维，不包括最近的obstacle和wall的坐标，配套`test_safe_env_backup_11-30.py`

### 9. control_car.py
> 小车控制代码