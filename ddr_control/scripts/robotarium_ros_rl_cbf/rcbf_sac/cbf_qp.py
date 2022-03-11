import numpy as np
from quadprog import solve_qp

class CascadeCBFLayer:

    def __init__(self, env, gamma_b=100, k_d=1.5, l_p=0.03):
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

        self.env = env
        self.u_min, self.u_max = self.get_control_bounds()
        self.gamma_b = gamma_b
        self.k_d = k_d
        self.l_p = l_p

    def get_u_safe(self, u_nom, s,other_s, mean_pred=None, sigma=None):
        """Given the current state of the system, this function computes the control input necessary to render the nominal
        control `u_nom` safe (i.e. u_safe + u_nom is safe).

        Parameters
        ----------
        u_nom : ndarray
            Nominal control input.
        s : ndarray
           current state.
        mean_pred : ndarray
            mean prediction.
        sigma : ndarray
            standard deviation in additive disturbance.

        Returns
        -------
        u_safe : ndarray
            Safe control input to be added to `u_nom` as such `env.step(u_nom + u_safe)`
        """

        P, q, G, h = self.get_cbf_qp_constraints(u_nom, s,other_s, mean_pred, sigma)
        u_safe = self.solve_qp(P, q, G, h)

        return u_safe

    def get_cbf_qp_constraints(self, u_nom, state, other_state, mean_pred=None, sigma_pred=None):
        """Build up matrices required to solve qp
        Program specifically solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Each control barrier certificate is of the form:
            dh/dx^T (f_out + g_out u) >= -gamma^b h_out^3 where out here is an output of the state.

        In the case of SafetyGym_point dynamics:
        state = [x y θ v ω]
        state_d = [v*cos(θ) v*sin(θ) omega ω u^v u^ω]

        Parameters
        ----------
        u_nom : ndarray
            Nominal control input.
        state : ndarray
            current state (check dynamics.py for details on each dynamics' specifics)
        mean_pred : ndarray
            mean disturbance prediction state, dimensions (n_s, dim_u)
        sigma_pred : ndarray
            standard deviation in additive disturbance after undergoing the output dynamics.

        Returns
        -------
        P : ndarray
            Quadratic cost matrix in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        q : ndarray
            Linear cost vector in qp (minimize_{u,eps} 0.5 * u^T P u + q^T u)
        G : ndarray
            Inequality constraint matrix (G[u,eps] <= h) of size (num_constraints, dim_u + 1)
        h : ndarray
            Inequality constraint vector (G[u,eps] <= h) of size (num_constraints,)
        """
        # print(other_state)
        collision_radius = self.env.hazards_radius  

        # p(x): lookahead output
        p_x = np.array([state[0] + self.l_p * np.cos(state[2]), state[1] + self.l_p * np.sin(state[2])])

        # p_dot = f_p + g_p u  where f_p(x) = 0,  g_p(x) = RL 
        f_p = np.zeros(2)
        theta = state[2]
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        R = np.array([[c_theta, -s_theta],
                        [s_theta, c_theta]])
        L = np.array([[1, 0],
                        [0, self.l_p]])
        g_p = R @ L

        # hs
        hs = 0.5 * (np.sum((p_x - other_state) ** 2,
                                    axis=1) - collision_radius ** 2)  # 1/2 * (||x - x_obs||^2 - r^2)
        dhdxs = (p_x - other_state)  # each row is dhdx_i for hazard i

        # since we're dealing with p(x), we need to get the disturbance on p_x, p_y
        # Mean
        # mean_p = mean_pred[:2] + self.l_p * np.array([-np.sin(state[2]), np.cos(state[2])]) * mean_pred[2]
        # # Sigma (output is 2-dims (p_x, p_y))
        # sigma_p = sigma_pred[:2] + self.l_p * np.array([-np.sin(state[2]), np.cos(state[2])]) * sigma_pred[2]

        dim_u = u_nom.shape[0]  # dimension of control inputs
        num_constraints = hs.shape[0] + 2 * dim_u  # each cbf is a constraint, and we need to add actuator constraints (dim_u of them)

        # Inequality constraints (G[u, eps] <= h)
        G = np.zeros((num_constraints, dim_u + 1))  # the plus 1 is for epsilon (to make sure qp is always feasible)
        h = np.zeros(num_constraints)
        ineq_constraint_counter = 0

        # First let's add the cbf constraints
        for c in range(hs.shape[0]):
            # extract current affine cbf (h = h1^T x + h0)
            h_ = hs[c]
            dhdx_ = dhdxs[c]

            # Add inequality constraints
            G[ineq_constraint_counter, :dim_u] = -dhdx_ @ g_p  # h1^Tg(x)
            G[ineq_constraint_counter, dim_u] = -1  # for slack
            
            # h[ineq_constraint_counter] = self.gamma_b * (h_ ** 3) + np.dot(dhdx_, (f_p+mean_p)) + np.dot(dhdx_ @ g_p,
            #                                                                                         u_nom) \
            #                                 - self.k_d * np.dot(np.abs(dhdx_), sigma_p)
            h[ineq_constraint_counter] = self.gamma_b * (h_ ** 3)  + np.dot(dhdx_ @ g_p,u_nom)

            ineq_constraint_counter += 1

        # Let's also build the cost matrices, vectors to minimize control effort and penalize slack
        P = np.diag([1.e1, 1.e-4, 1e5])  # in the original code, they use 1e24 instead of 1e7, but quadprog can't handle that...
        q = np.zeros(dim_u + 1)


        # Second let's add actuator constraints
        dim_u = u_nom.shape[0]  # dimension of control inputs
        for c in range(dim_u):

            # u_max >= u_nom + u ---> u <= u_max - u_nom
            if self.u_max is not None:
                G[ineq_constraint_counter, c] = 1
                h[ineq_constraint_counter] = self.u_max[c] - u_nom[c]
                ineq_constraint_counter += 1

            # u_min <= u_nom + u ---> -u <= u_min - u_nom
            if self.u_min is not None:
                G[ineq_constraint_counter, c] = -1
                h[ineq_constraint_counter] = -self.u_min[c] + u_nom[c]
                ineq_constraint_counter += 1
        return P, q, G, h

    def solve_qp(self, P, q, G, h):
        """Solves:
            minimize_{u,eps} 0.5 * u^T P u + q^T u
                subject to G[u,eps]^T <= h

        Parameters
        ----------
        P : ndarray
            Quadratic cost matrix
        q : ndarray
            Linear cost vector
        G : ndarray
            Inequality Constraint Matrix
        h : ndarray
            Inequality Constraint Vector

        Returns
        -------
        u_safe : ndarray
            The solution of the qp without the last dimension (the slack).
        """

        # print('P =\t {}'.format(P))
        # print('q =\t {}'.format(q))
        # print('G =\t {}'.format(G))
        # print('h =\t {}'.format(h))

        # Here we normalize G and h to stay consistent with what we do in CVXPYLAYER which often crashes with big #s
        Gh = np.concatenate((G, np.expand_dims(h, 1)), 1)
        Gh_norm = np.expand_dims(np.max(np.abs(Gh), axis=1), axis=1)
        G /= Gh_norm
        h = h / Gh_norm.squeeze(-1)

        try:
            sol = solve_qp(P, q, -G.T, -h)
            u_safe = sol[0][:-1]
            print('quadprog = {} eps = {}'.format(u_safe, sol[0][-1]))
        except ValueError as e:
            print('P = {},\nq = {},\nG = {},\nh = {}.'.format(P, q, G, h))
            raise e

        if np.abs(sol[0][-1]) > 1e-1:
            print('CBF indicates constraint violation might occur. epsilon = {}'.format(sol[0][-1]))

        return u_safe

    def get_cbfs(self, hazards_locations, hazards_radius):
        """Returns CBF function h(x) and its derivative dh/dx(x) for each hazard. Note that the CBF is defined with
        with respect to an output of the state.

        Parameters
        ----------
        hazards_locations : list
            List of hazard-xy positions where each item is a list of length 2
        hazards_radius : float
            Radius of the hazards

        Returns
        -------
        get_h :
            List of cbfs each corresponding to a constraint. Each cbf is affine (h(s) = h1^Ts + h0), as such each cbf is represented
            as a tuple of two entries: h_cbf = (h1, h0) where h1 and h0 are ndarrays.
        get_dhdx :
        """

        hazards_locations = np.array(hazards_locations)
        collision_radius = hazards_radius + 0.07  # add a little buffer

        def get_h(state):
            # p(x): lookahead output
            state = np.array([state[0] + self.l_p * np.cos(state[2]), state[1] + self.l_p * np.sin(state[2])])
            return 0.5 * (np.sum((state - hazards_locations)**2, axis=1) - collision_radius**2)  # 1/2 * (||x - x_obs||^2 - r^2)

        def get_dhdx(state):
            # p(x): lookahead output
            
            state = np.array([state[0] + self.l_p * np.cos(state[2]), state[1] + self.l_p * np.sin(state[2])])
            dhdx = (state - hazards_locations)  # each row is dhdx_i for hazard i
            return dhdx

        return get_h, get_dhdx

    def get_control_bounds(self):
        """

        Returns
        -------
        u_min : ndarray
            min control input.
        u_max : ndarray
            max control input.
        """

        u_min = self.env.u_min
        u_max = self.env.u_max

        return u_min, u_max

    def get_min_h_val(self, state,other_state):
        """

        Parameters
        ----------
        state : ndarray
            Current State

        Returns
        -------
        min_h_val : float
            Minimum h(x) over all hs. If below 0, then at least 1 constraint is violated.

        """

        get_h, _ = self.get_cbfs(other_state, self.env.hazards_radius)
        min_h_val = np.min(get_h(state))
        return min_h_val

