# 运行说明

## TD3

### 训练时运行

```shell
 python main_td3.py --mode train --cuda --device_num 0 --max_episodes 3500 --model_number 001 --use_cbf --log_comet  
```

### 测试：

```shell
python main.py --mode test  --seed 133 --cuda --device_num 3 --model_number 20220418 --use_cbf --time_delay --disturbance
```

> --model_number: 为选择的模型序号
> --use_cbf 为使用cbf；若不使用cbf，则不带这个参数即可。
> --device_num 1 represents the device number of the cuda.
> --time_delay 是否加延迟
> --disturbance 是否加测试干扰
> --ros_env 是否使用ros环境（前提是开启了相关的ros gazebo 环境）


```shell
python main.py --mode test  --seed 133 --cuda --device_num 3 --model_number 2022042302 --use_cbf --time_delay --disturbance
```

## SAC

### 训练时运行

```shell
python main_sac.py --cuda --device_num 0 --use_cbf  --eval --max_episodes 1000 --log_comet   
```

> --use_cbf 为使用cbf；若不使用cbf，则不带这个参数即可。
> --eval 表示每训练50个episodes测试一次。
> --device_num 1 represents the device number of the cuda.
> --time_delay 是否加延迟
> --disturbance 是否加测试干扰
> --ros_env 是否使用ros环境（前提是开启了相关的ros gazebo 环境）


### 测试时

```shell
python main_sac.py --cuda --use_cbf --seed 12345 --mode test --resume ./output/name  --visualize
```

如：

```shell
python main_sac.py --cuda --device_num 1 --seed 12345 --use_cbf  --mode test --resume ./output/Robotarium-run23  --visualize
```

- 有延迟时

    ```shell
    python main_sac.py --cuda --device_num 1 --seed 12345 --use_cbf --time_delay --mode test --resume ./output/Robotarium-run23  --visualize
    ```

- 有干扰时
    ```shell
    python main_sac.py --cuda --device_num 1 --seed 12345 --use_cbf --disturbance --mode test --resume ./output/Robotarium-run23  --visualize
    ```


- 既有延迟又有干扰时
    ```shell
    python main_sac.py --cuda --device_num 1 --seed 12345 --use_cbf --disturbance --time_delay  --mode test --resume ./output/Robotarium-run23  --visualize
    ```


最好的效果是Robotarium-run23