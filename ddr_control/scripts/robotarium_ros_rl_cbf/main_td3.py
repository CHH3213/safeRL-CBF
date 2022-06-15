# import comet_ml at the top of your file
from comet_ml import Experiment
import numpy as np
import torch
import argparse
import os

from td3 import TD3
import sys
from td3.train import train
from td3.eval import eval
from td3.utils import prYellow
sys.path.append(os.path.join(os.path.dirname(__file__), '../new_envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append('..')
from controllers.simple_pv import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Comet ML
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    parser.add_argument('--comet_key', default='Kp7gLz2KdgLb9KBjDqrgVvrhz', help='Comet API key')
    parser.add_argument('--comet_workspace', default='chh3213', help='Comet workspace')
    # env and train related
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3)
    parser.add_argument("--env", default="ros_robotarium_env")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=256, type=int)  # Time steps initial random policy is used
    parser.add_argument("--save_freq", default=50, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=5, type=int, help='how many episode to perform during validate experiment')  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--max_episodes", default=1000, type=int)  # Max time steps to run environment
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    # algo settings related
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',help='hidden size (default: 256)')
    # model related
    parser.add_argument("--model_number", default=0, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", default=True, action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model",default="default")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--render', action="store_true", help='Should the display be used.')
    parser.add_argument('--restore',  action="store_true", help='Should the compensator be used.')
    
    
    parser.add_argument("--agent_number", default=3, type=int, help ="number of agents") # number of agents
    parser.add_argument('--time_delay', action="store_true", help='set the time delaym default:0.')
    parser.add_argument('--disturbance', action="store_true", help='Should the disturbance be added.')
    parser.add_argument('--use_cbf', action="store_true", help='Should the cbf be used.')
    parser.add_argument('--k_d', default=3, type=float)
    parser.add_argument('--gamma_b', default=200, type=float)
    parser.add_argument('--l_p', default=0.15, type=float, help="Look-ahead distance for unicycle dynamics output.")
    parser.add_argument('--ros_env', action="store_true", help='Should the ros gazebo environment be used.')

    args = parser.parse_args()

    file_name = f"{args.policy}_cbf_{args.use_cbf}_model_{args.model_number}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")
    if args.log_comet:
        prYellow('Logging experiment on comet.ml!')

    if not os.path.exists(f"./td3/results/{file_name}"):
        os.makedirs(f"./td3/results/{file_name}")

    if args.save_model and not os.path.exists(f"./td3/models/{file_name}"):
        os.makedirs(f"./td3/models/{file_name}")

    # Environment
    print('use cbf:', args.use_cbf)
    if args.mode == "train":
        from new_envs.rps.rl_env.robotarium_env import RobotariumEnv  # cbf train used
    else:
        if args.ros_env:
            from new_envs.rl_ros_robotarium_env import RobotariumEnv
            # from new_envs.rl_ros_robotarium_env_multi_target import RobotariumEnv
        else:
            from new_envs.rps.rl_env.robotarium_env_multi_target import RobotariumEnv  # cbf test used

    env = RobotariumEnv(args)
    # Random Seed
    if args.seed > 0:
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "args":args,
        "env":env,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    if args.cuda:
        torch.cuda.set_device(args.device_num)
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)

    if args.mode == "train":
        train(env, args, policy, file_name)
    else:
        if env.agent_number==10:
            # index_list = [0, 2, 3, 5, 6, 8, 9]  # 原来只有两个目标点来回
            index_list = [i for i in range(8)]  # 10 agents
        elif env.agent_number==3:
            index_list = [0]
        # index_list = [0, 1, 2, 6, 7, 8, 9]
        # index_list = [0, 1, 2, 3,4,5,6, 7, 8, 9]
        # index_list = [0, 1, 2, 3, 4, 5, 6]
        # index_list = [0, 1]
        for idx in index_list:
            assert idx < env.agent_number and idx >= 0, "idex must be less than number of agents"
        eval(env, args, policy, file_name, index_list)
