import argparse
import numpy as np
import torch
from rcbf_sac.dynamics import DYNAMICS_MODE
from rcbf_sac.utils import to_tensor, prRed, prCyan
from time import time
from qpth.qp import QPFunction


class CBFQPLayer:

    def __init__(self, env, args, gamma_b=100, k_d=1.5, l_p=0.03):
        """Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        gamma_b : float, optional
            gamma of control barrier certificate.
        k_d : float, optional
            confidence parameter desired (2.0 corresponds to ~95% for example).
        """

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b

        # todo: env中要有agent_number属性
        self.num_cbfs = len(env.hazards_locations) + env.agent_number - 1  ## cbf的数量为其他智能体数量+障碍物数量
        self.k_d = k_d
        self.l_p = l_p

        self.action_dim = env.action_space.shape[0]
        self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim  # 不等式约束包括cbf约束和控制输入约束

    def get_safe_action(self, state_batch, other_state_batch, action_batch, mean_pred_batch=None, sigma_batch=None):
        """

        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        other_state_batch: torch.tensor or ndarray  为其他智能体与障碍物的位置状态，不包括角度，即shape为[batch_size,2]
        action_batch : torch.tensor or ndarray
            State batch
        mean_pred_batch : torch.tensor or ndarray
            Mean of disturbance
        sigma_batch : torch.tensor or ndarray
            Standard deviation of disturbance

        Returns
        -------
        final_action_batch : torch.tensor
            Safe actions to take in the environment.
        """

        # batch form if only a single data point is passed
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            action_batch = action_batch.unsqueeze(0)
            state_batch = state_batch.unsqueeze(0)
            mean_pred_batch = mean_pred_batch.unsqueeze(0)
            sigma_batch = sigma_batch.unsqueeze(0)

        start_time = time()
        Ps, qs, Gs, hs = self.get_cbf_qp_constraints(state_batch, other_state_batch, action_batch, mean_pred_batch,
                                                     sigma_batch)
        build_qp_time = time()
        safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
        # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
        # The actual safe action is the cbf action + the nominal action
        final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1),
                                   self.u_max.repeat(action_batch.shape[0], 1))

        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, dim_u+1, dim_u+1)
        qs : torch.Tensor
            (batch_size, dim_u+1)
        Gs : torch.Tensor
            (batch_size, num_ineq_constraints, dim_u+1)
        hs : torch.Tensor
            (batch_size, num_ineq_constraints)
        Returns
        -------
        safe_action_batch : torch.tensor
            The solution of the qp without the last dimension (the slack).
        """

        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)
        sol = self.cbf_layer(Ps, qs, Gs, hs,
                             solver_args={"check_Q_spd": False, "maxIter": 100000, "notImprovedLim": 10, "eps": 1e-4})
        safe_action_batch = sol[:, :-1]
        return safe_action_batch

    def cbf_layer(self, Qs, ps, Gs, hs, As=None, bs=None, solver_args=None):
        """

        Parameters
        ----------
        Qs : torch.Tensor
        ps : torch.Tensor
        Gs : torch.Tensor
            shape (batch_size, num_ineq_constraints, num_vars)
        hs : torch.Tensor
            shape (batch_size, num_ineq_constraints)
        As : torch.Tensor, optional
        bs : torch.Tensor, optional
        solver_args : dict, optional

        Returns
        -------
        result : torch.Tensor
            Result of QP
        """

        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=0, **solver_args)(Qs.double(), ps.double(), Gs.double(), hs.double(), As,
                                                      bs).float()

        if torch.any(torch.isnan(result)):
            prRed('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result

    def get_cbf_qp_constraints(self, state_batch, other_state_batch, action_batch, mean_pred_batch=None,
                               sigma_pred_batch=None):
        """Build up matrices required to solve qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        In the case of SafetyGym_point dynamics:
        state = [x y θ v ω]
        state_d = [v*cos(θ) v*sin(θ) omega ω u^v u^ω]

        Quick Note on batch matrix multiplication for matrices A and B:
            - Batch size should be first dim
            - Everything needs to be 3-dimensional
            - E.g. if B is a vec, i.e. shape (batch_size, vec_length) --> .view(batch_size, vec_length, 1)

        Parameters
        ----------
        state_batch : torch.tensor
            current state (check dynamics.py for details on each dynamics' specifics)
        action_batch : torch.tensor
            Nominal control input.
        mean_pred_batch : torch.tensor
            mean disturbance prediction state, dimensions (n_s, dim_u)
        sigma_pred_batch : torch.tensor
            standard deviation in additive disturbance after undergoing the output dynamics.
        gamma_b : float, optional
            CBF parameter for the class-Kappa function

        Returns
        -------
        P : torch.tensor
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : torch.tensor
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : torch.tensor
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, dim_u + 1)
        h : torch.tensor
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """

        # assert len(state_batch.shape) == 2 and len(action_batch.shape) == 2 and len(mean_pred_batch.shape) == 2 and len(
        #     sigma_pred_batch.shape) == 2, print(state_batch.shape, action_batch.shape, mean_pred_batch.shape,
        #                                         sigma_pred_batch.shape)

        batch_size = state_batch.shape[0]
        gamma_b = self.gamma_b

        # Expand dims shape:[batch_size,7,1]
        state_batch = torch.unsqueeze(state_batch, -1)
        other_state_batch = torch.unsqueeze(other_state_batch, -1)
        action_batch = torch.unsqueeze(action_batch, -1)
        # mean_pred_batch = torch.unsqueeze(mean_pred_batch, -1)
        # sigma_pred_batch = torch.unsqueeze(sigma_pred_batch, -1)

        num_cbfs = self.num_cbfs
        hazards_radius = self.env.hazards_radius
        # hazards_locations = to_tensor(self.env.hazards_locations, torch.FloatTensor, self.device)
        collision_radius = 1.3 * hazards_radius  # add a little buffer
        l_p = self.l_p

        thetas = state_batch[:, 2, :].squeeze(-1)  # shape:([256])
        c_thetas = torch.cos(thetas)
        s_thetas = torch.sin(thetas)

        # p(x): lookahead output (batch_size, 2)
        ps = torch.zeros((batch_size, 2)).to(self.device)
        ps[:, 0] = state_batch[:, 0, :].squeeze(-1) + l_p * c_thetas
        ps[:, 1] = state_batch[:, 1, :].squeeze(-1) + l_p * s_thetas

        # p_dot(x) = f_p(x) + g_p(x)u  where f_p(x) = 0,  g_p(x) = RL 

        # f_p(x) = [0,...,0]^T
        f_ps = torch.zeros((batch_size, 2, 1)).to(self.device)

        # g_p(x) = RL where L = diag([1, l_p])
        Rs = torch.zeros((batch_size, 2, 2)).to(self.device)  # shape([batch_size,2,2])
        Rs[:, 0, 0] = c_thetas
        Rs[:, 0, 1] = -s_thetas
        Rs[:, 1, 0] = s_thetas
        Rs[:, 1, 1] = c_thetas
        Ls = torch.zeros((batch_size, 2, 2)).to(self.device)
        Ls[:, 0, 0] = 1
        Ls[:, 1, 1] = l_p
        # bmm pytorch tensor相乘 p*m*n,p*n*l-->p*m*l
        g_ps = torch.bmm(Rs, Ls)  # (batch_size, 2, 2)  

        # D_p(x) = g_p [0 D_θ]^T + [D_x1 D_x2]^T
        # mu_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
        # mu_theta_aug[:, 1, :] = mean_pred_batch[:, 2, :]
        # mu_ps = torch.bmm(g_ps, mu_theta_aug) + mean_pred_batch[:, :2, :]
        # sigma_theta_aug = torch.zeros([batch_size, 2, 1]).to(self.device)
        # sigma_theta_aug[:, 1, :] = sigma_pred_batch[:, 2, :]
        # sigma_ps = torch.bmm(torch.abs(g_ps), sigma_theta_aug) + sigma_pred_batch[:, :2, :]

        # hs (batch_size, hazards_locations)
        ps_hzds = ps.repeat((1, num_cbfs)).reshape((batch_size, num_cbfs, 2))

        # print('p',np.shape(ps_hzds))
        # print('state',np.shape(other_state_batch.view(batch_size, num_cbfs, -1)))
        # print(ps_hzds - other_state_batch.view(batch_size, num_cbfs, -1))
        hs = 0.5 * (torch.sum((ps_hzds - other_state_batch.view(batch_size, num_cbfs, -1)) ** 2,
                              axis=2) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)

        dhdps = (ps_hzds - other_state_batch.view(batch_size, num_cbfs, -1))  # (batch_size, n_cbfs, 2)
        # (batch_size, 2, 1)
        dim_u = action_batch.shape[1]  # dimension of control inputs
        num_constraints = num_cbfs + 2 * dim_u  # each cbf is a constraint, and we need to add actuator constraints (dim_u of them)

        # Inequality constraints (G[u, eps] <= h)
        G = torch.zeros((batch_size, num_constraints, dim_u + 1)).to(
            self.device)  # the extra variable is for epsilon (to make sure qp is always feasible)
        h = torch.zeros((batch_size, num_constraints)).to(self.device)
        ineq_constraint_counter = 0

        # Add inequality constraints
        G[:, :num_cbfs, :dim_u] = -torch.bmm(dhdps, g_ps)  # h1^Tg(x)
        G[:, :num_cbfs, dim_u] = -1  # for slack
        # h[:, :num_cbfs] = gamma_b * (hs ** 3) + (torch.bmm(dhdps, f_ps + mu_ps) - torch.bmm(torch.abs(dhdps), sigma_ps) + torch.bmm(torch.bmm(dhdps, g_ps), action_batch)).squeeze(-1)
        h[:, :num_cbfs] = gamma_b * (hs ** 3) + (
                torch.bmm(dhdps, f_ps) + torch.bmm(torch.bmm(dhdps, g_ps), action_batch)).squeeze(-1)
        ineq_constraint_counter += num_cbfs

        # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
        P = torch.diag(torch.tensor([1.e0, 1.e-2, 1e5])).repeat(batch_size, 1, 1).to(self.device)
        q = torch.zeros((batch_size, dim_u + 1)).to(self.device)

        # Second let's add actuator constraints
        dim_u = action_batch.shape[1]  # dimension of control inputs

        for c in range(dim_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[:, ineq_constraint_counter, c] = 1
                h[:, ineq_constraint_counter] = self.u_max[c] - action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[:, ineq_constraint_counter, c] = -1
                h[:, ineq_constraint_counter] = -self.u_min[c] + action_batch[:, c].squeeze(-1)
                ineq_constraint_counter += 1

        return P, q, G, h

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : torch.tensor
            min control input.
        u_max : torch.tensor
            max control input.
        """

        u_min = torch.tensor(self.env.safe_action_space.low).to(self.device)
        u_max = torch.tensor(self.env.safe_action_space.high).to(self.device)

        return u_min, u_max


if __name__ == "__main__":

    from build_env import build_env
    from rcbf_sac.dynamics import DynamicsModel
    from copy import deepcopy
    from rcbf_sac.utils import to_numpy, prGreen


    def simple_controller(env, state, goal):
        goal_xy = goal[:2]
        goal_dist = -np.log(goal[2])  # the observation is np.exp(-goal_dist)
        v = 0.02 * goal_dist
        relative_theta = 1.0 * np.arctan2(goal_xy[1], goal_xy[0])
        omega = 1.0 * relative_theta

        return np.clip(np.array([v, omega]), env.action_space.low, env.action_space.high)


    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="SafetyGym", help='Options are Unicycle or SafetyGym')
    parser.add_argument('--robot_xml', default='xmls/point.xml',
                        help="SafetyGym Currently only supporting xmls/point.xml")
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=100, type=float)
    parser.add_argument('--l_p', default=0.03, type=float)
    parser.add_argument('--gp_model_size', default=2000, type=int, help='gp')
    parser.add_argument('--cuda', action='store_true', help='run on CUDA (default: False)')
    args = parser.parse_args()
    # Environment
    env = build_env(args)

    device = torch.device('cuda' if args.cuda else 'cpu')


    def to_def_tensor(ndarray):

        return to_tensor(ndarray, torch.FloatTensor, device)


    diff_cbf_layer = CBFQPLayer(env, args, args.gamma_b, args.k_d, args.l_p)
    dynamics_model = DynamicsModel(env, args)

    obs = env.reset()
    done = False

    ep_ret = 0
    ep_cost = 0
    ep_step = 0

    for i_step in range(3000):

        if done:
            prGreen('Episode Return: %.3f \t Episode Cost: %.3f' % (ep_ret, ep_cost))
            ep_ret, ep_cost, ep_step = 0, 0, 0
            obs = env.reset()

        state = dynamics_model.get_state(obs)

        print('state = {}, dist2hazards = {}'.format(state[:2],
                                                     np.sqrt(np.sum((env.hazards_locations - state[:2]) ** 2, 1))))

        disturb_mean, disturb_std = dynamics_model.predict_disturbance(state)

        action = simple_controller(env, state, obs[-3:])  # TODO: observations last 3 indicated
        # action = 2*np.random.rand(2) - 1.0
        assert env.action_space.contains(action)
        final_action = diff_cbf_layer.get_safe_action(to_def_tensor(state), to_def_tensor(action),
                                                      to_def_tensor(disturb_mean), to_def_tensor(disturb_std))
        final_action = to_numpy(final_action)

        # Env Step
        observation2, reward, done, info = env.step(final_action)
        observation2 = deepcopy(observation2)

        # Update state and store transition for GP model learning
        next_state = dynamics_model.get_state(observation2)
        if ep_step % 2 == 0:
            dynamics_model.append_transition(state, final_action, next_state)

        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_step += 1
        # env.render()

        obs = observation2
        state = next_state
