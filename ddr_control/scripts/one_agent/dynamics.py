import numpy as np
import torch
from gp_model import GPyDisturbanceEstimator
from utils import to_tensor, to_numpy

"""
This file contains the DynamicsModel class. Depending on the environment selected, it contains the dynamics
priors and learns the remaining disturbance term using GPs. The system is described as follows: 
                        x_dot ∈ f(x) + g(x)u + D(x), 
where x is the state, f and g are the drift and control dynamics respectively, and D(x) is the learned disturbance set. 

A few things to note:
    - The prior depends on the dynamics. This is hard-coded as of now. In the future, one direction to take would be to 
    get a few episodes in the env to determine the affine prior first using least-squares or something similar.
    - The state is not the same as the observation and typically requires some pre-processing. These functions are supplied
    here which are also, unfortunately, hard-coded as of now. 
    - The functions here are sometimes used in batch-form or single point form, and sometimes the inputs are torch tensors 
    and other times are numpy arrays. These are things to be mindful of, and output format is always same as input format.
"""

DYNAMICS_MODE = {'Unicycle': {'n_s': 3, 'n_u': 2},  # state = [x y θ]
                 'SimulatedCars': {'n_s': 10, 'n_u': 1}}  # state = [x y θ v ω]
MAX_STD = {'Unicycle': [2e-1, 2e-1, 2e-1], 'SimulatedCars': [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2]}


class DynamicsModel:

    def __init__(self, env, args):
        """Constructor of DynamicsModel.

        Parameters
        ----------
        env : gym.env
            Gym environment.
        """

        self.env = env
        # Get Dynamics
        self.get_f, self.get_g = self.get_dynamics()
        self.n_s = DYNAMICS_MODE[self.env.dynamics_mode]['n_s']
        self.n_u = DYNAMICS_MODE[self.env.dynamics_mode]['n_u']

        # Keep Disturbance History to estimate it using GPs
        self.disturb_estimators = None
        self.disturbance_history = dict()
        self.history_counter = 0  # keeping only 1000 points in the buffer
        self.max_history_count = args.gp_model_size  # How many points we want to have in the GP
        self.disturbance_history['state'] = np.zeros((self.max_history_count, self.n_s))
        self.disturbance_history['disturbance'] = np.zeros((self.max_history_count, self.n_s))
        self.train_x = None  # x-data used to fit the last GP models
        self.train_y = None  # y-data used to fit the last GP models

        # Point Robot specific dynamics (approx using unicycle + look-ahead)
        if hasattr(args, 'l_p'):
            self.l_p = args.l_p

        self.device = torch.device("cuda" if args.cuda else "cpu")

    def predict_next_state(self, state_batch, u_batch, t_batch=None):
        """Given the current state and action, this function predicts the next state.

        Parameters
        ----------
        state_batch : ndarray
            State
        u_batch : ndarray
            Action
        t_batch: ndarray, optional
            Time batch for state dependant dynamics

        Returns
        -------
        next_state : ndarray
            Next state
        """

        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = np.expand_dims(state_batch, axis=0)

        # Start with our prior for continuous time system x' = f(x) + g(x)u
        if t_batch is not None:
            next_state_batch = state_batch + self.env.dt * (self.get_f(state_batch, t_batch) + (
                    self.get_g(state_batch, t_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1))
        else:
            next_state_batch = state_batch + self.env.dt * (
                    self.get_f(state_batch) + (self.get_g(state_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1))

        pred_std = np.zeros(state_batch.shape)

        if expand_dims:
            next_state_batch = next_state_batch.squeeze(0)
            if pred_std is not None:
                pred_std = pred_std.squeeze(0)

        if t_batch is not None:
            next_t_batch = t_batch + self.env.dt
            return next_state_batch, self.env.dt * pred_std, next_t_batch

        return next_state_batch, self.env.dt * pred_std, t_batch

    def predict_next_obs(self, state, u):
        """Predicts the next observation given the state and u. Note that this only predicts the mean next observation.

        Parameters
        ----------
        state : ndarray
        u : ndarray

        Returns
        -------
        next_obs : ndarray
            Next observation
        """

        next_state, _, _ = self.predict_next_state(state, u)
        next_obs = self.get_obs(next_state)
        return next_obs

    def get_dynamics(self):
        """Get affine CBFs for a given environment.

        Parameters
        ----------

        Returns
        -------
        get_f : callable
                Drift dynamics of the continuous system x' = f(x) + g(x)u
        get_g : callable
                Control dynamics of the continuous system x' = f(x) + g(x)u
        """

        if self.env.dynamics_mode == 'Unicycle':

            def get_f(state_batch, t_batch=None):
                f_x = np.zeros(state_batch.shape)
                return f_x

            def get_g(state_batch, t_batch=None):
                theta = state_batch[:, 2]
                g_x = np.zeros((state_batch.shape[0], 3, 2))
                g_x[:, 0, 0] = np.cos(theta)
                g_x[:, 1, 0] = np.sin(theta)
                g_x[:, 2, 1] = 1.0
                return g_x


        else:
            raise Exception('Unknown Dynamics mode.')

        return get_f, get_g

    def get_state(self, obs):
        """Given the observation, this function does the pre-processing necessary and returns the state.

        Parameters
        ----------
        obs_batch : ndarray or torch.tensor
            Environment observation.

        Returns
        -------
        state_batch : ndarray or torch.tensor
            State of the system.

        """

        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)

        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = to_numpy(obs)

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        if self.env.dynamics_mode == 'Unicycle':
            theta = np.arctan2(obs[:, 3], obs[:, 2])
            state_batch = np.zeros((obs.shape[0], 3))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta
        else:
            raise Exception('Unknown dynamics')

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return to_tensor(state_batch, dtype, device) if is_tensor else state_batch

    def get_obs(self, state_batch):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------
        state : ndarray
            Environment state batch of shape (batch_size, n_s)

        Returns
        -------
        obs : ndarray
          Observation batch of shape (batch_size, n_o)

        """

        if self.env.dynamics_mode == 'Unicycle':
            obs = np.zeros((state_batch.shape[0], 4))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])
            obs[:, 3] = np.sin(state_batch[:, 2])
        else:
            raise Exception('Unknown dynamics')
        return obs

    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
