import argparse
import numpy as np
import torch
from sac.utils import to_tensor, prRed, prCyan
from time import time
from qpth.qp import QPFunction

"""
add slack variable
"""
class CBFQPLayer:

    def __init__(self, env, args, Vp,Ve,r,K1,K2):
        """
        Constructor of CBFLayer.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        args : parameters

        Vp: pursuer's velocity

        Ve: evader's velocity

        r: safety critical  distance

        K1, K2: cbf constraiants

        """

        self.device = torch.device(f"cuda:{args.device_num}" if args.cuda else "cpu")

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.Vp = Vp

        self.num_cbfs = 3 
        self.Ve = Ve
        self.r = r
        self.K1 = K1
        self.K2 = K2

        self.action_dim = env.action_space.shape[0]
        self.num_ineq_constraints = self.num_cbfs + 2 * self.action_dim  # all ineq_constraints includes cbf constraints and control input constraints

    def get_safe_action(self, state_batch, other_state_batch, action_batch):
        """

        Parameters
        ----------
        state_batch : torch.tensor or ndarray
        other_state_batch: torch.tensor or ndarray  为其他智能体与障碍物的位置状态，不包括角度，即shape为[batch_size,2]
        action_batch : torch.tensor or ndarray


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
            other_state_batch = other_state_batch.unsqueeze(0)
           
        # print("state_batch.shape",len(state_batch.shape))
        start_time = time()
        Ps, qs, Gs, hs = self.get_cbf_qp_constraints(state_batch, other_state_batch, action_batch)
        build_qp_time = time()
        safe_action_batch = self.solve_qp(Ps, qs, Gs, hs)
        # prCyan('Time to get constraints = {} - Time to solve QP = {} - time per qp = {} - batch_size = {} - device = {}'.format(build_qp_time - start_time, time() - build_qp_time, (time() - build_qp_time) / safe_action_batch.shape[0], Ps.shape[0], Ps.device))
        # The actual safe action is the cbf action + the nominal action
        # print('a', action_batch)
        # print('s', safe_action_batch)
        # print(self.u_max)
        # final_action = torch.clamp(action_batch + safe_action_batch, self.u_min.repeat(action_batch.shape[0], 1),
        #                            self.u_max.repeat(action_batch.shape[0], 1))
        final_action = safe_action_batch
        # print(action_batch,safe_action_batch)
        return final_action if not expand_dims else final_action.squeeze(0)

    def solve_qp(self, Ps: torch.Tensor, qs: torch.Tensor, Gs: torch.Tensor, hs: torch.Tensor):
        """求解qp:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        Ps : torch.Tensor
            (batch_size, dim_u)
        qs : torch.Tensor
            (batch_size, dim_u)
        Gs : torch.Tensor
            (batch_size, num_ineq_constraints, dim_u+1)
        hs : torch.Tensor
            (batch_size, num_ineq_constraints)
        Returns
        -------
        safe_action_batch : torch.tensor
            The solution of the qp without the last dimension (the slack).
        """
        # print(np.shape(qs))
        Ghs = torch.cat((Gs, hs.unsqueeze(2)), -1)
        Ghs_norm = torch.max(torch.abs(Ghs), dim=2, keepdim=True)[0]
        Gs /= Ghs_norm
        hs = hs / Ghs_norm.squeeze(-1)

        sol = self.cbf_layer(Ps, qs, Gs, hs, solver_args={"check_Q_spd": False, "maxIter": 10000, "notImprovedLim": 10, "eps": 1e-4})
        safe_action_batch = sol[:, :-1]
        # print(sol)
        # print(safe_action_batch)
        # print(np.shape(safe_action_batch))
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
        # TODO: fix bug

        if solver_args is None:
            solver_args = {}

        if As is None or bs is None:
            As = torch.Tensor().to(self.device).double()
            bs = torch.Tensor().to(self.device).double()

        result = QPFunction(verbose=0, **solver_args)(Qs.double(), ps.double(), Gs.double(), hs.double(), As, bs).float()

        if torch.any(torch.isnan(result)):
            prRed('QP Failed to solve - result is nan == {}!'.format(torch.any(torch.isnan(result))))
            raise Exception('QP Failed to solve')
        return result

    def get_cbf_qp_constraints(self, state_batch, other_state_batch, action_batch):
        """
        minimize_{u,eps} 0.5 * u^T P u + q^T u
            subject to G[u,eps]^T <= h
        Parameters
        ----------
        state_batch : torch.tensor
            current state 
        action_batch : torch.tensor
            Nominal control input.


        Returns
        -------
        P : torch.tensor
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + Q^T u)
        Q : torch.tensor
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + Q^T u)
        G : torch.tensor
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, dim_u + 1)
        H : torch.tensor
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        
        """
        # DONE: 检查P,Q,G,H的维度是否符合
        batch_size = state_batch.shape[0] 
        # state_batch = torch.unsqueeze(state_batch, -1) # self agent's state
        # other_state_batch = torch.unsqueeze(other_state_batch, -1)  # nearest two agents and nearest obstacle'state
        # action_batch = torch.unsqueeze(action_batch, -1)
        num_cbfs = other_state_batch.shape[1]

        dim_u = action_batch.shape[1]  # dimension of control inputs

        G = torch.zeros((batch_size, self.num_ineq_constraints,dim_u+1)).to(self.device) # G
        H = torch.zeros((batch_size, self.num_ineq_constraints)).to(self.device)
        g, h = self._cbf_constraints(state_batch, other_state_batch, batch_size) # g:[batchsize,num_cbfs]
        ineq_constraint_counter = num_cbfs

        
        G[:, :num_cbfs,:dim_u] = g.unsqueeze(-1)
        G[:, :num_cbfs, dim_u] = -1  # for slack
        # print(np.shape(g.unsqueeze(-1)))

        H[:, :num_cbfs] =h
        for c in range (dim_u):
            G[:, ineq_constraint_counter,c] = 1
            # H[:, ineq_constraint_counter] = self.u_max- action_batch[:, c].squeeze(-1)
            H[:, ineq_constraint_counter] = self.u_max
            ineq_constraint_counter +=1
            G[:, ineq_constraint_counter,c] = -1
            # H[:, ineq_constraint_counter] = -self.u_min+ action_batch[:, c].squeeze(-1)
            H[:, ineq_constraint_counter] = -self.u_min
            ineq_constraint_counter +=1

        P = torch.diag(torch.tensor([1.e-4,1.e4])).repeat(batch_size,1, 1).to(self.device)
        Q = torch.zeros((batch_size, dim_u + 1)).to(self.device)
        Q[:,:dim_u] = -2*action_batch
        # print(np.shape(G),np.shape(H))

        return P,Q,G,H
    
    def _cbf_constraints(self, state_batch, other_state_batch,batch_size):
        # print(np.shape(other_state_batch))
        x_p = other_state_batch[:,:,0]
        y_p = other_state_batch[:,:,1]
        # P_p = other_state_batch[:,:,0:2]
        theta_p = other_state_batch[:,:,2]
        cos_p = torch.cos(theta_p)
        sin_p = torch.sin(theta_p)

        x_e = state_batch[:,0]
        y_e = state_batch[:,1]
        # x_e = x_e.unsqueeze(-1)
        # y_e = y_e.unsqueeze(-1)

        # P_e = state_batch[:,0:2]
        theta_e = state_batch[:,2]
        cos_e = torch.cos(theta_e)
        sin_e = torch.sin(theta_e)  
        # cos_e = cos_e.unsqueeze(-1)
        # sin_e = sin_e.unsqueeze(-1)


        dim_num_cbf = other_state_batch.shape[1]

        # print(dim_num_cbf)
        # print(np.shape(theta_e),np.shape(theta_p))
        # print(np.shape(cos_p))
        # print(np.shape(cos_e))
        # print(np.shape(cos_e.repeat(1, dim_num_cbf).reshape((batch_size, dim_num_cbf))))
        # print(cos_e)
        # print(cos_e.repeat(1, dim_num_cbf))
        delta_v = torch.zeros((batch_size, dim_num_cbf,2)).to(self.device)   
        delta_v[:,:,0] = self.Vp*cos_p-self.Ve*cos_e.unsqueeze(-1).repeat(1, dim_num_cbf).to(self.device)
        delta_v[:,:,1] = self.Vp*sin_p-self.Ve*sin_e.unsqueeze(-1).repeat(1, dim_num_cbf).to(self.device)
        # print(np.shape(x_p))
        # print(np.shape(x_e))
        delta_p = torch.zeros((batch_size,dim_num_cbf, 2)).to(self.device)   
        delta_p[:,:,0]=x_p-x_e.unsqueeze(-1).repeat(1, dim_num_cbf).to(self.device)
        delta_p[:,:,1]=y_p-y_e.unsqueeze(-1).repeat(1, dim_num_cbf).to(self.device)
        # print(np.shape(delta_p))
        ve_dot=torch.zeros((batch_size, 2)).to(self.device)  
        ve_dot[:,0]=-self.Ve*sin_e
        ve_dot[:,1]=-self.Ve*cos_e
        # print(np.shape(delta_p))
        # print(np.shape(ve_dot))
        # print(delta_p)
        # print(delta_v)
        # print(torch.sum(delta_p*delta_v,axis=2))

        # 已检查下面维度和计算均正确
        g = 2*torch.bmm(delta_p,ve_dot.unsqueeze(-1))
        # print(np.shape(g))
        # print(np.shape(theta_e),np.shape(theta_p))
        # print(np.shape(torch.cos(theta_p-theta_e)))
        # print(np.shape(delta_p),np.shape(delta_v))
        # print(np.shape(torch.sum(delta_p**2,axis=2)-self.r**2))
        # print(np.shape(torch.sum(delta_p*delta_v,axis=2)))
        h = 2*(self.Vp**2+self.Ve**2-2*self.Vp*self.Ve*torch.cos(theta_p-theta_e.unsqueeze(-1).repeat(1, dim_num_cbf)))+\
        self.K1*(torch.sum(delta_p**2,axis=2)-self.r**2)+2*self.K2*torch.sum(delta_p*delta_v,axis=2)
        # print(np.shape(g.squeeze(-1)))
        # print(np.shape(h))
        # print(g)
        # print(g.squeeze(-1))
        return g.squeeze(-1),h





        

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
