> ## 对于logger小车而言：
>

1. 角速度为负,逆时针转动，为正,顺时针转动
2. 姿态角向左为正，向右为负

## envs文件说明

### 1. robotarium_get.py

> 用于模型的状态设置，初始化目标点等。

### 2. rotate_calculation.py

> 姿态旋转相关函数

### 9. control_car.py

> 小车控制代码

### ros_robotarium_env.py

是clf-cbf-mpc代码的ros运行环境,
related environment: robotarium.world
related launch file:robotarium.launch

### rl_ros_robotarium_env.py

是ros_rl_cbf代码的ros运行环境
related environment: robotarium_ros_rl.world
related launch file:robotarium_ros_rl.launch

