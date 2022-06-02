# robotarium_env

使用CBF，奖励函数不考虑动态障碍物。

# robotarium_env_nocbf

不使用CBF，奖励函数考虑将失控小车作为动态障碍物。

# robotarium_env_multi_target

多个目标点测试环境

```python
if self.agent_number == 10:
a_0 = np.array([0.5 * np.random.rand() + 3, 0.5 *
                np.random.rand() + 3, -1.57 * np.random.rand() - 1.57])
a_1 = np.array([a_0[0]-0.5, a_0[1]-0.5, 6.28 * np.random.rand() - 3.14])
a_2 = np.array([0.5 * np.random.rand() - 3,  0.5*
                np.random.rand() + 3, -1.57 * np.random.rand()])
a_3 = np.array([a_2[0]+0.5, a_2[1]-0.5, 6.28 * np.random.rand() - 3.14])
a_4 = np.array([0.5 * np.random.rand() - 2.5, 0.5 * np.random.rand() - 3., 1.57 * np.random.rand()])
a_5 = np.array([a_4[0]+0.5, a_4[1]+0.5, 6.28 * np.random.rand() - 3.14])
a_6 = np.array([0.5 * np.random.rand() + 3, 0.5 * np.random.rand() - 3., 1.57 * np.random.rand() +1.57])
a_7 = np.array([a_6[0]+0.5, a_6[1]-0.5, 6.28 * np.random.rand() - 3.14])
a_8 = np.array([-1 * np.random.rand(), 2 *
                    np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
    a_9 = np.array([1 * np.random.rand(), 2 *
                    np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])


    # self.initial_conditions = np.array(
    #     [a_0, a_0-0.75, a_2, a_2+0.75, a_4, a_4+0.75, a_6, a_6+0.75, a_8, a_9]).T
    self.initial_conditions = np.array(
        [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9]).T
    
elif self.agent_number == 3:
    a_0 = np.array([0.5 * np.random.rand() + 3, 0.5 *
                    np.random.rand() + 3, -1.57 * np.random.rand() - 1.57])
    a_1 = np.array([a_0[0]-0.5, a_0[1]-0.5, 6.28 * np.random.rand() - 3.14])
    a_2 = np.array([2 * np.random.rand() - 1, 2 *
                    np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
    self.initial_conditions = np.array([a_0, a_1, a_2]).T
    # self.initial_conditions = np.array([[0, 0.5, 0], [0, 1, 0], [0, -1, 0]]).T

```




重新训练新的环境，在新的测试中状态量不包括自己的位置  目标点[3，3]
```python
    a_0 = np.array([0.5 * np.random.rand() + 3, 0.5 *
                    np.random.rand() + 3, -1.57 * np.random.rand() - 1.57])
    a_1 = np.array([a_0[0]+0.5, a_0[1]+0.5, 6.28 * np.random.rand() - 3.14])
    a_2 = np.array([0.5 * np.random.rand() - 3.5,  0.5*
                    np.random.rand() + 3, -1.57 * np.random.rand()])
    a_3 = np.array([a_2[0]-0.5, a_2[1]+0.5, 6.28 * np.random.rand() - 3.14])
    a_4 = np.array([0.5 * np.random.rand() - 3.5, 0.5 * np.random.rand() - 3.5, 1.57 * np.random.rand()])
    a_5 = np.array([a_4[0]-0.5, a_4[1]-0.5, 6.28 * np.random.rand() - 3.14])
    a_6 = np.array([0.5 * np.random.rand() + 3, 0.5 * np.random.rand() - 3.5, 1.57 * np.random.rand() +1.57])
    a_7 = np.array([a_6[0]+0.5, a_6[1]-0.5, 6.28 * np.random.rand() - 3.14])
    a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
    a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
      
    # self.initial_conditions = np.array(
    #     [a_0, a_0-0.75, a_2, a_2+0.75, a_4, a_4+0.75, a_6, a_6+0.75, a_8, a_9]).T
    self.initial_conditions = np.array(
        [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9]).T

```



轨迹：
```python
                画轨迹用
                a_0, a_1 = np.array([1,2,0]),np.array([3,2,-3.14])
                a_2, a_3 = np.array([-1,2,-3.14]),np.array([-3,2,-3.14])
                a_4, a_5 = np.array([-1,-2,-3.14]),np.array([-3,-2,0])
                a_6, a_7 = np.array([1,-2,-3.14]),np.array([3,-2,-3.14])
                a_8, a_9 = np.array([-0.5,0,-3.14]),np.array([0.5,0,0])
```


备个份--未使用
```python
                x1 = 2 * np.random.rand() + 1
                x1 = x1+1 if x1 >= 2 else x1
                a_0 = np.array([x1, x1, 6.28 * np.random.rand() - 3.14])
                a_1 = np.array([a_0[0]+0.5, a_0[1]+0.5, 6.28 * np.random.rand() - 3.14])
                x2 = -2 * np.random.rand() - 1
                x2 = x2-1 if x2 <=-2 else x2
                a_2 = np.array([x2,  x1, 6.28 * np.random.rand() - 3.14])
                a_3 = np.array([a_2[0]-0.5, a_2[1]-0.5, 6.28 * np.random.rand() - 3.14])
                x3 = -2 * np.random.rand() - 1
                x3 = x3-1 if x3 <=-2 else x3
                a_4 = np.array([x3, x3, 6.28 * np.random.rand() - 3.14])
                a_5 = np.array([a_4[0]-0.5, a_4[1]-0.5, 6.28 * np.random.rand() - 3.14])

                a_6 = np.array([x1, x3, 6.28 * np.random.rand() - 3.14])
                a_7 = np.array([a_6[0]+1, a_6[1]-0.5, 6.28 * np.random.rand() - 3.14])
                a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
                a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
                
```


20220419-19-27 
目标点[4,4],大范围随机
```python

        a_0 = np.array([1.5 * np.random.rand()+2.5, 1.5 * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])
        a_1 = np.array([1.5 * np.random.rand()+1, 1.5 * np.random.rand()+1, 6.28 * np.random.rand() - 3.14])

        a_2 = np.array([-1.5 * np.random.rand()-2.5, 1.5 * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])
        a_3 = np.array([-1.5 * np.random.rand()-1, 1.5 * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])

        a_4 = np.array([-1.5 * np.random.rand()-2.5, -1.5 * np.random.rand()-2.5, 6.28 * np.random.rand() - 3.14])
        a_5 = np.array([-1.5 * np.random.rand()-1, -1.5 * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])

        a_6 = np.array([1.5 * np.random.rand()+2.5, -1.5 * np.random.rand()-2.5, 6.28 * np.random.rand() - 3.14])
        a_7 = np.array([1.5 * np.random.rand()+1, -1.5 * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])
        a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
        a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
```

目标点[4,4],小范围随机
```python

    a_0 = np.array([1.5 * np.random.rand()+2.5, 1.5 * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])
    a_1 = np.array([1.5 * np.random.rand()+1, 1.5 * np.random.rand()+1, 6.28 * np.random.rand() - 3.14])

    a_2 = np.array([-1.5 * np.random.rand()-2.5, 1.5 * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])
    a_3 = np.array([-1.5 * np.random.rand()-1, 1.5 * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])

    a_4 = np.array([-1.5 * np.random.rand()-2.5, -1.5 * np.random.rand()-2.5, 6.28 * np.random.rand() - 3.14])
    a_5 = np.array([-1.5 * np.random.rand()-1, -1.5 * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])

    a_6 = np.array([1.5 * np.random.rand()+2.5, -1.5 * np.random.rand()-2.5, 6.28 * np.random.rand() - 3.14])
    a_7 = np.array([1.5 * np.random.rand()+1, -1.5 * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])
    a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
    a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
```


目标点[3,3]大范围随机

```python
    a_0 = np.array([1. * np.random.rand()+2., 1. * np.random.rand()+2., 6.28 * np.random.rand() - 3.14])
    a_1 = np.array([1. * np.random.rand()+1, 1. * np.random.rand()+1, 6.28 * np.random.rand() - 3.14])

    a_2 = np.array([-1. * np.random.rand()-2., 1. * np.random.rand()+2., 6.28 * np.random.rand() - 3.14])
    a_3 = np.array([-1. * np.random.rand()-1, 1. * np.random.rand()+2., 6.28 * np.random.rand() - 3.14])

    a_4 = np.array([-1. * np.random.rand()-2., -1. * np.random.rand()-2., 6.28 * np.random.rand() - 3.14])
    a_5 = np.array([-1. * np.random.rand()-1, -1. * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])

    a_6 = np.array([1. * np.random.rand()+2., -1. * np.random.rand()-2., 6.28 * np.random.rand() - 3.14])
    a_7 = np.array([1. * np.random.rand()+1, -1. * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])
    a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
    a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
```



20220420 范围

各自目标点1的范围内随机
```python


                a_0 = np.array([1 * np.random.rand()+0.5, 1 * np.random.rand()+0.5, 6.28 * np.random.rand() - 3.14])
                a_1 = np.array([1. * np.random.rand()+2.5, 1. * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])

                a_2 = np.array([-1. * np.random.rand()-0.5, 1. * np.random.rand()+0.5, 6.28 * np.random.rand() - 3.14])
                a_3 = np.array([-1. * np.random.rand()-2.5, 1. * np.random.rand()+2.5, 6.28 * np.random.rand() - 3.14])

                a_4 = np.array([-1. * np.random.rand()-0.5, -1. * np.random.rand()-0.5, 6.28 * np.random.rand() - 3.14])
                a_5 = np.array([-1. * np.random.rand()-2.5, -1. * np.random.rand()-2.5, 6.28 * np.random.rand() - 3.14])

                a_6 = np.array([1 * np.random.rand()+0.5, -1. * np.random.rand()-0.5, 6.28 * np.random.rand() - 3.14])
                a_7 = np.array([1 * np.random.rand()+2.5, -1. * np.random.rand()-2.5, 6.28 * np.random.rand() - 3.14])
                
                a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
                a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
```

有用的范围

```python

        a_0 = np.array([0.5 * np.random.rand()+2.5, 0.5 * np.random.rand()+1, 6.28 * np.random.rand() - 3.14])
        a_1 = np.array([0.5 * np.random.rand()+2.5, 0.5 * np.random.rand()+3.5, 6.28 * np.random.rand() - 3.14])

        a_2 = np.array([-0.5 * np.random.rand()-2.5, 0.5 * np.random.rand()+1, 6.28 * np.random.rand() - 3.14])
        a_3 = np.array([-0.5 * np.random.rand()-2.5, 0.5 * np.random.rand()+3.5, 6.28 * np.random.rand() - 3.14])

        a_4 = np.array([-0.5 * np.random.rand()-2.5, -0.5 * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])
        a_5 = np.array([-0.5 * np.random.rand()-2.5, -0.5 * np.random.rand()-3.5, 6.28 * np.random.rand() - 3.14])

        a_6 = np.array([0.5 * np.random.rand()+2.5, -0.5 * np.random.rand()-1, 6.28 * np.random.rand() - 3.14])
        a_7 = np.array([0.5 * np.random.rand()+2.5, -0.5 * np.random.rand()-3.5, 6.28 * np.random.rand() - 3.14])
        
        a_8 = np.array([-1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])
        a_9 = np.array([1 * np.random.rand(), 2 * np.random.rand() - 1, 6.28 * np.random.rand() - 3.14])

```




```python

                if i == 0:
                    x1, y1 = 1 * np.random.rand()+0.5, 1 * np.random.rand()+0.5
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0

                elif i == 1:
                    x1, y1 = 1. * np.random.rand()+2.5, 1. * np.random.rand()+2.5
                    # x1, y1 = 0, 1
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 2:
                    x1, y1 = -1. * np.random.rand()-0.5, 1. * np.random.rand()+0.5
                    # x1, y1 = 0, -1
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 3:
                    x1, y1 = -1. * np.random.rand()-2.5, 1. * np.random.rand()+2.5
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 4:
                    x1, y1 = -1. * np.random.rand()-0.5, -1. * np.random.rand()-0.5
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 5:
                    x1, y1 = -1. * np.random.rand()-2.5, -1. * np.random.rand()-2.5
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 6:
                    x1, y1 = 1 * np.random.rand()+0.5, -1. * np.random.rand()-0.5
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 7:
                    x1, y1 = 1 * np.random.rand()+2.5, -1. * np.random.rand()-2.5
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 8:
                    x1, y1 = -0.5 * np.random.rand(), 2 * np.random.rand() - 1
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0
                elif i == 9:
                    x1, y1 = 0.5 * np.random.rand(), 2 * np.random.rand() - 1
                    my_model.model_state.pose.position.x = x1
                    my_model.model_state.pose.position.y = y1
                    my_model.model_state.pose.position.z = 0

```