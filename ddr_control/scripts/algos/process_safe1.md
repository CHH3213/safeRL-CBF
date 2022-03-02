
    ```
    def _compute_reward(self, agent_index, target):
        """
        Compute reward and done based on current status
        :param agent_index: 智能体索引
        :param target: 目标点位置(x,y)
        Return:
            reward
            done
        """
        # rospy.logdebug("\nStart Computing Reward")
        # 到目标点的距离
        dist_agent2target = np.linalg.norm(
            np.array(self.obs_dict[agent_index]['position']) - np.array(target))
        reward, done = 0, False
        if agent_index == 0 or agent_index == 1:
            if dist_agent2target < 0.5:
                reward += 300
                if agent_index == 0 or agent_index == 1:
                    self.is_success[agent_index] = True
                    # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                done = True
    
            reward += -2.0 * dist_agent2target
    
        # 各个智能体间的距离
        # for d in self.obs_dict[agent_index]['delta_dist']:
        #     if d < 0.51:
        #         reward -= 1000
        #         # done = True
        #         self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
    
        #     else:
        #         reward += 0
        
        for i, d in enumerate(self.obs_dict[agent_index]['delta_dist']):
            if d < 0.51:
                reward -= 100
                done = True
                # if agent_index == 0 or agent_index == 1:
                #     self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
    
            # elif agent_index == 0 or 1 and i != 2:     # modify :if good agents between bad agent, reward +=d
            #     reward += 0.2*d
            # elif agent_index == 1 and i == 2:     # modify :if good agents between bad agent, reward +=d
            #     reward += -0.5*1/d
            # elif agent_index == 0 and i == 2:
            #     reward += -0.5*1/d
            # else:
            #     reward += 0.2*d
        # rospy.logdebug("\nEnd Computing Reward\n")
    
        # 碰到障碍物奖励
        # for barrier in self.barrier_name:
        #     [barrier_position, _, _, _, _] = self.gazebo_reset.get_state(barrier)
        #     dist_agent2barrier = np.linalg.norm(
        #         np.array(self.obs_dict[agent_index]['position']) - np.array(barrier_position[0:2]))
        #     if dist_agent2barrier < 0.5:
        #         reward -= 1000
        #         self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
        #         # done = True
    
        # 碰到墙奖励
        for wall in self.walls_name:
            [wall_position, _, _, _, _] = self.gazebo_reset.get_state(wall)
            if wall == 'wall_0' or wall == 'wall_1':
                dist_agent2wall = np.abs(self.obs_dict[agent_index]['position'][1] - wall_position[1])
            else:
                dist_agent2wall = np.abs(self.obs_dict[agent_index]['position'][0] - wall_position[0])
            if agent_index == 0 or agent_index == 1 or agent_index == 2:
            
                if dist_agent2wall < 0.4:
                    reward -= 100
                    # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                    done = True
                else:  # add reward of wall
                    reward += 0
                    if agent_index == 1 or agent_index == 0:
                        reward += -0.4 * 1/dist_agent2wall
    
        return reward, done


    def _safe_constrains(self, good_obs, bad_obs,good_action,bad_action):
        '''
        good_obs: good agent's state
        bad_obs: bad agent's state
        good_action: good agent's action
        bad_action: bad agent's action
        '''
        Vp = copy.deepcopy(self.v_linear)
        Ve = copy.deepcopy(self.v_linear)
        r = self.safe_radius 
        Xe, Ye, th_e = good_obs[0],good_obs[1],good_obs[2]
        Xp, Yp, th_p = bad_obs[0],bad_obs[1],bad_obs[2]
        sinp, cosp, sine, cose = np.sin(th_p), np.cos(th_p), np.sin(th_e), np.cos(th_e)
        Dp = np.array([Xp - Xe, Yp - Ye])
        Dv = np.array([Vp * cosp - Ve * cose, Vp * sinp - Ve * sine])
        Dot_vp = np.array([-Vp * sinp, Vp * cosp])
        Dot_ve = np.array([-Ve * sine, Ve * cose])
    
        # TODO: 根据h的实时情况调整k0,k1,
        p1, p2 = 2, 2
        K1 = p1 + p2
        K0 = p1*p2
    
        _safe_cons = 2 * (Vp ** 2 + Ve ** 2 - 2 * Vp * Ve * np.cos(th_p - th_e)) + \
            2 * bad_action * np.einsum('i,i->', Dp, Dot_vp) -\
            2 * good_action * np.einsum('i,i->', Dp, Dot_ve) + \
            2 * K1 * (np.einsum('i,i->', Dp, Dv)) +\
            K0 * (np.einsum('i,i->', Dp, Dp) - r ** 2)
        # _safe_cons = np.sum(np.square(Dp))-r**2
        return _safe_cons



    def _compute_safe(self, good_obs, bad_obs, good_action, bad_action):
        """
        Compute safe value based on current status for all good agents
        Return:
            safe_value  效果类比于 reward
        """
        h = self._safe_constrains(good_obs, bad_obs, good_action, bad_action) # CBF
        # TODO: safe value的表达需要再调整
        safe_value =-np.exp(-h)
        if h < 0: 
            safe_value -= 100
    
        return safe_value
        ```


        td3_safe_GOOD_2021-11-21-16:52:17

------

2021/11/24  9:15

------

direct CBF, reach target, works

models:`td3_safe_2021-11-23-23:35:20`

safe state is not good.



------

**nanorobot angular velocity direction is on the contraly**

td3_only_nod ------------ no distance to others and no CBFs

reward:

```python
    def _compute_reward(self, agent_index, target):
        """
        Compute reward and done based on current status
        :param agent_index: 智能体索引
        :param target: 目标点位置(x,y)
        Return:
            reward
            done
        """
        # rospy.logdebug("\nStart Computing Reward")
        # 到目标点的距离
        dist_agent2target = np.linalg.norm(
            np.array(self.obs_dict[agent_index]['position']) - np.array(target))
        reward, done = 0, False
        if agent_index == 0 or agent_index == 1:
            if dist_agent2target < 0.2:
                # reward += 1000  # before 2021/11/24/11:53
                reward += 1000
                if agent_index == 0 or agent_index == 1:
                    self.is_success[agent_index] = True
                    # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                done = True

            # reward += -2.0 * dist_agent2target  # before 2021/11/24/11:53
            reward += -2.0 * dist_agent2target  
            # if dist_agent2target < 1.415:
            #     reward += 1 / dist_agent2target 
        
        # for pos in self.obs_dict[agent_index]['position']:
        #     if np.abs(pos)>3:
        #         # self.gazebo_reset.reset_agent_state(self.ddr_list)
        #         reward -= 500
        #         done = True
        

        # 各个智能体间的距离
        # for d in self.obs_dict[agent_index]['delta_dist']:
        #     if d < 0.51:
        #         reward -= 1000
        #         # done = True
        #         self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])

        #     else:
        #         reward += 0

        for i, d in enumerate(self.obs_dict[agent_index]['delta_dist']):
            if d < 0.25:
                reward -= 500
                # reward -= 100  # before 2021/11/24/11:53
                done = True
                # if agent_index == 0 or agent_index == 1:
                #     self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])

            # elif agent_index == 0 or 1 and i != 2:     # modify :if good agents between bad agent, reward +=d
            #     reward += 0.2*d
            # elif agent_index == 1 and i == 2:     # modify :if good agents between bad agent, reward +=d
            #     reward += -0.5*1/d
            # elif agent_index == 0 and i == 2:
            #     reward += -0.5*1/d
            # else:
            #     reward += 0.2*d
        # rospy.logdebug("\nEnd Computing Reward\n")

        # 碰到障碍物奖励
        for barrier in self.barrier_name:
            [barrier_position, _, _, _, _] = self.gazebo_reset.get_state(barrier)
            dist_agent2barrier = np.linalg.norm(
                np.array(self.obs_dict[agent_index]['position']) - np.array(barrier_position[0:2]))
            if dist_agent2barrier < 0.25:
                reward -= 500
                # reward -= 100  # before 2021/11/24/11:53
                # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                done = True

        # 碰到墙奖励  # after 2021/11/25/9:33, been used
        for wall in self.walls_name:
            [wall_position, _, _, _, _] = self.gazebo_reset.get_state(wall)
            if wall == 'wall_0' or wall == 'wall_1':
                dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][1] - wall_position[1])
            else:
                dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][0] - wall_position[0])
            if agent_index == 0 or agent_index == 1 or agent_index == 2:

                if dist_agent2wall < 0.25:
                    reward -= 500
                #self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                    done = True
                # else:  # add reward of wall
                #     reward += 0
                #     if agent_index == 1 or agent_index == 0:
                #         # reward += -0.4 * 1/dist_agent2wall  # 
                #         reward += -0.1 * 1/dist_agent2wall

        return reward, done

```

------

2021-11-29-0:40

------

td3_nocbf_2021-11-28-11:11:03

-->

td3_nocbf_2021-11-28-12:03:08

-->

td3_nocbf_2021-11-29-00:40:45
