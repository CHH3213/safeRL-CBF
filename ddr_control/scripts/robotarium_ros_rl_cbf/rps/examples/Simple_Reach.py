import numpy as np
def Simple_Catcher(attacker_state,defender_state):
    is_poistive = 1
    distance = np.sqrt(np.sum(np.square(attacker_state[:2] - defender_state[:2])))
    dx =attacker_state[0] - defender_state[0]
    dy =attacker_state[1] - defender_state[1]
    theta_e = np.arctan2(dy,dx) - defender_state[2]
    # print(np.arctan(dy/dx))
    # print(defender_state[2])
    # attacker_state[2] - defender_state[2]
    if(theta_e>np.pi):
        theta_e -= 2*np.pi
    elif (theta_e<-np.pi):
        theta_e += 2*np.pi
    # print(theta_e)
    
    if(theta_e>np.pi/2 or theta_e<-np.pi/2):
        is_poistive = -1

    u = 1.5*distance*is_poistive
    w = theta_e*is_poistive
    u = np.clip(u, -0.32, 0.32)
    w = np.clip(w, -2, 2)
    return np.array([u,w])



