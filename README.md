# Safe RL using DDR Robots in Gazebo Simulation

- 任务：将CBF结合进actor网络中，然后再通过critic更新


## Environments

- 将整个IEEE_TAI_DDR功能包放到ros工作空间中，并编译和source

  ```bash
  cd ~/ros_ws/
  catkin_make
  source devel/setup.sh
  ```

- **将ddr_gazebo/models下的model文件全部放到home目录下：~/.gazebo/models/**

- 运行

  ```bash
  roslaunch ddr_control robotarium.launch
  ```
  或者：
  ```bash
  roslaunch ddr_control robotarium_rl.launch
  ```
  打开环境，正常打开后应该会有如下界面：

  <img src="./ddr_gazebo/worlds/2022-03-11_20-04.png" alt="world" style="zoom:150%;" />

- 目录`ddr_control/scripts/envs`下有两个环境文件，其中`ros_robotarium_env`环境用来测试clf-cbf-mpc，`rl_ros_robotarium_env`用来测试rl+CBF算法。

## algos

### 1. clf_cbf_nmpc_test

进入`clf_cbf_nmpc_test`文件夹，运行`test.py`文件，则直接测试clf-cbf-mpc算法。

### 2. robotarium_ros_rl_cbf

进入`robotarium_ros_rl_cbf`文件夹

- `sac`文件夹存储的是sac算法相关的文件
- `main_sac.py`是sac算法训练测试的主函数
- `td3`文件夹存储的是td3算法相关的文件
- `main_td3.py`是sac算法训练测试的主函数
- `cbf_qp_layer.py`是没有加松弛变量下的、仅有角速度控制量的QP layer代码
- `cbf_qp_layer_bk.py`是加松弛变量下的、仅有角速度控制量的QP layer代码

其余请详细查看文件夹内的readme文档