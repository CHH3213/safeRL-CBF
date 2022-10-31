# import comet_ml at the top of your file

from comet_ml import Experiment

import argparse
import torch
import numpy as np

from sac.sac import SAC
import os

from sac.utils import prGreen, get_output_folder, prYellow
from sac.evaluator import Evaluator
from sac.evaluator import test
from sac.train import train
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../new_envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append('..')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="Robotarium", help='Options: Robotarium.')
    # Comet ML
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    parser.add_argument('--comet_key', default='Kp7gLz2KdgLb9KBjDqrgVvrhz', help='Comet API key')
    parser.add_argument('--comet_workspace', default='chh3213', help='Comet workspace')
    # SAC Args
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')



    parser.add_argument('--restore', type=bool, default=False, help='Should the compensator be used.')
    parser.add_argument('--visualize', action='store_true', dest='visualize',
                        help='visualize env -only in available test mode')
    parser.add_argument('--output', default='./sac/output', type=str, help='')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', action="store_true",
                        help='Evaluates a policy a policy every 5 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_episodes', type=int, default=400, metavar='N',
                        help='maximum number of episodes (default: 400)')
    parser.add_argument('--save_frenquency', type=int, default=20, metavar='N',
                        help='save frenquency of training (default: 20)')
    
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--validate_episodes', default=100, type=int,
                        help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int,
                        help='how many steps to perform a validate experiment')
    parser.add_argument('--render', action="store_true",
                        help='Should the display be used.')

    # 对比实验设置
    # parser.add_argument('--time_delay', type=float, default=0., metavar='G', help='set the time delaym default:0.')
    parser.add_argument('--time_delay', action="store_true", help='set the time delaym default:0.')
    parser.add_argument('--disturbance', action="store_true", help='Should the disturbance be added.')
    parser.add_argument('--ros_env', action="store_true", help='Should the ros gazebo environment be used.')
    parser.add_argument("--agent_number", default=3, type=int, help ="number of agents") # number of agents

    ## CBF
    parser.add_argument('--use_cbf', action="store_true", help='Should the cbf be used.')
    parser.add_argument('--k_d', default=1.5, type=float)
    parser.add_argument('--gamma_b', default=200, type=float, help="gamma of control barrier condition")
    parser.add_argument('--l_p', default=0.15, type=float, help="Look-ahead distance for robotarium output.")

    args = parser.parse_args()

    index = 0

    if args.mode == 'train':
        args.output = get_output_folder(args.output, args.env_name)
    if args.resume == 'default':
        args.resume =  './sac/output/{}-run0'.format(args.env_name)
    elif args.resume.isnumeric():
        args.resume =  './sac/output/{}-run{}'.format(args.env_name, args.resume)

    # if args.cuda:
    #     torch.cuda.set_device(args.device_num)

    if args.mode == 'train' and args.log_comet:
        if args.use_cbf:
            project_name = 'cbf-layer-environment-sac'
        else:
            project_name = 'no_cbf-layer-environment-sac'
        prYellow('Logging experiment on comet.ml!')
        # Create an experiment with your api key
        experiment = Experiment(
            api_key=args.comet_key,
            project_name=project_name,
            workspace=args.comet_workspace,
        )
        # Log args on comet.ml
        experiment.log_parameters(vars(args))

        experiment_tags = [str(args.batch_size) + '_batch']

        print(experiment_tags)
        experiment.add_tags(experiment_tags)
    else:
        experiment = None

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
    if args.mode == 'train' and args.log_comet:
        experiment.log_parameter('obs_dim',env.observation_space.shape[0])

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, env, args)

    # Random Seed
    if args.seed > 0:
        # env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.mode == 'train':

        if args.restore:
            agent.load_weights(args.resume)
        train(agent, env, args, experiment, index)
    elif args.mode == 'test':
        if env.agent_number==10:
            # index_list = [0, 1,2, 3, 5, 6, 8, 9]  # 原来只有两个目标点来回
            index_list = [i for i in range(8)]  # 10 agents
        elif env.agent_number==3:
            index_list = [0, 1]
        # index_list = [0, 1, 2, 6, 7, 8, 9]
        # index_list = [0, 1, 2, 3,4,5,6, 7, 8, 9]
        # index_list = [0, 1, 2, 3, 4, 5, 6]
        # index_list = [0, 1]
        for idx in index_list:
            assert idx < env.agent_number and idx >= 0, "idex must be less than number of agents"
        evaluate = Evaluator(args, args.validate_episodes, args.validate_steps, args.seed, args.use_cbf, args.resume)
        test(agent, env, index_list, evaluate, args.resume, visualize=args.visualize, debug=True)
