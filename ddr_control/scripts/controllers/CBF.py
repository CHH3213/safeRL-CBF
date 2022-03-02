import numpy as np
from scipy.optimize import minimize


def QPfun(ud):
    def fun(u):
        return (u[0] - ud[0]) ** 2 / 2
    return fun


def constrains(State):
    '''
    State[0] = Xp
    State[1] = Yp
    State[2] = th_p (rad)
    State[3] = Xe
    State[4] = Ye
    State[5] = th_e (rad)
    State[6] = wp
    '''
    Vp = 0.15
    Ve = 0.15
    Vgood = 0.15
    r = 0.35

    Xp, Yp, th_p, Xe, Ye, th_e, wp,Xgood,Ygood,th_good,w_good,Xb,Yb,Xw,Yw = State
    ''' pursuer-evader'''
    sinp, cosp, sine, cose = np.sin(th_p), np.cos(th_p), np.sin(th_e), np.cos(th_e)
    pv_Dp = np.array([Xp - Xe, Yp - Ye])
    pv_Dv = np.array([Vp * cosp - Ve * cose, Vp * sinp - Ve * sine])
    pv_Dot_vp = np.array([-Vp * sinp, Vp * cosp])
    pv_Dot_ve = np.array([-Ve * sine, Ve * cose])
    # TODO：调整K0，K1的值
    pv_p1, pv_p2 = 1.5,1.5
    pv_K1 = pv_p1 + pv_p2
    pv_K0 = pv_p1 * pv_p2
    # pv_K1 = 0.03
    # pv_K0 = 2.5
    def con_we1(we):
        return 2 * (Vp ** 2 + Ve ** 2 - 2 * Vp * Ve * np.cos(th_p - th_e)) + 2 * wp * np.einsum('i,i->', pv_Dp, pv_Dot_vp) - \
            2 * we * np.einsum('i,i->', pv_Dp, pv_Dot_ve) + \
            pv_K0 * (np.einsum('i,i->', pv_Dp, pv_Dp) - r ** 2) + \
            2 * pv_K1 * (np.einsum('i,i->', pv_Dp, pv_Dv))
    '''good - good'''
    sin_good, cos_good, sine, cose = np.sin(th_good), np.cos(th_good), np.sin(th_e), np.cos(th_e)

    g_Dp = np.array([Xgood - Xe, Ygood - Ye])
    g_Dv = np.array([Vgood * cos_good - Ve * cose, Vgood * sin_good - Ve * sine])
    g_Dot_vp = np.array([-Vgood * sin_good, Vgood * cos_good])
    g_Dot_ve = np.array([-Ve * sine, Ve * cose])
    # TODO：调整K0，K1的值
    g_p1, g_p2 = 1.5, 1.5
    g_K1 = g_p1 + g_p2
    g_K0 = g_p1 * g_p2
    # g_K1 = 0.03
    # g_K0 = 2.5
    def con_we2(we):
        return 2 * (Vgood ** 2 + Ve ** 2 - 2 * Vgood * Ve * np.cos(th_good - th_e)) + 2 * w_good * np.einsum('i,i->', g_Dp, g_Dot_vp) - \
            2 * we * np.einsum('i,i->', g_Dp, g_Dot_ve) + \
            g_K0 * (np.einsum('i,i->', g_Dp, g_Dp) - r ** 2) + \
            2 * g_K1 * (np.einsum('i,i->', g_Dp, g_Dv))

    '''good - barrier'''
    b_Dp = np.array([Xb - Xe, Yb - Ye])
    b_Dv = np.array([- Ve * cose, - Ve * sine])
    b_Dot_vp = np.array([0, 0])
    b_Dot_ve = np.array([-Ve * sine, Ve * cose])
    w_b = 0
    # TODO：调整K0，K1的值
    b_p1, b_p2 =1.5, 1.5
    b_K1 = b_p1 + b_p2
    b_K0 = b_p1 * b_p2
    # b_K1 = 0.03
    # b_K0 = 2.5
    def con_we3(we):
        return 2 * Ve ** 2 + 2 * w_b * np.einsum('i,i->', b_Dp, b_Dot_vp) - \
            2 * we * np.einsum('i,i->', b_Dp, b_Dot_ve) + \
            b_K0 * (np.einsum('i,i->', b_Dp, b_Dp) - r ** 2) + \
            2 * b_K1 * (np.einsum('i,i->', b_Dp, b_Dv))
    
    '''good - nearest wall'''
    Dp = np.array([Xw - Xe, Yw - Ye])
    Dv = np.array([- Ve * cose, - Ve * sine])
    Dot_vp = np.array([0, 0])
    Dot_ve = np.array([-Ve * sine, Ve * cose])
    w_w = 0
    # TODO：调整K0，K1的值
    p1, p2 = 1.5, 1.5
    K1 = p1 + p2
    K0 = p1 * p2
    # K1 = 0.03
    # K0 = 2.5
    r_wall = 0.35
    def con_we4(we):
        return 2 * Ve ** 2 + 2 * w_w * np.einsum('i,i->', Dp, Dot_vp) - \
            2 * we * np.einsum('i,i->', Dp, Dot_ve) + \
            K0 * (np.einsum('i,i->', Dp, Dp) - r_wall ** 2) + \
            2 * K1 * (np.einsum('i,i->', Dp, Dv))
    cons = (
        {'type': 'ineq', 'fun': con_we1},
        {'type': 'ineq', 'fun': con_we2},
        {'type': 'ineq', 'fun': con_we3},
        {'type': 'ineq', 'fun': con_we4},
        {'type': 'ineq', 'fun': lambda u: u[0] + np.pi},
        {'type': 'ineq', 'fun': lambda u: np.pi - u[0]}
    )

    return cons


def CBF(u, State):
    '''
    u=[we]
    -------pursuer--------
    State[0] = Xp
    State[1] = Yp
    State[2] = th_p (rad)
    -------good agent(self)
    State[3] = Xe
    State[4] = Ye
    State[5] = th_e (rad)
    -------pursuer speed----------
    State[6] = wp
    -------other good agent-----------
    State[7] = Xgood
    State[8] = Ygood
    State[9] = th_good (rad)
    State[10] = w_good
    ----------barrier---------
    State[11] = Xb
    State[12] = Yb
    -----------wall----------
    State[13] = Xw
    State[14] = Yw
    '''
    x0 = u
    # wmax, wmin = np.pi, -np.pi
    # if State[6] < 0:
    #     State[6] = wmax
    # elif State[6] > 0:
    #     State[6] = wmin
    # else:
    #     State[6] = 0

    res = minimize(fun=QPfun(u), x0=x0, constraints=constrains(State))
    # print(res.success)
    return res.x


if __name__ == '__main__':
    pass
