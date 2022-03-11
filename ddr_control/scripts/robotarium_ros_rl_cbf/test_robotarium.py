from numpy.random import gamma
import rps.robotarium as robotarium
# from rps.utilities.transformations import *
# from rps.utilities.barrier_certificates import *
# from rps.utilities.misc import *
# from rps.utilities.controllers import *
from rcbf_sac.cbf_qp import CascadeCBFLayer
import numpy as np
from matplotlib import patches

from rps.examples.Simple_Reach import Simple_Catcher


# N = 2
N = 5

a = np.array([[-1.5, -0.5 ,0]]).T
initial_conditions = a
for idx in range(1, N):
    initial_conditions = np.concatenate((initial_conditions, np.array([[2*np.random.rand()-1.0, 2.*np.random.rand()-1.0, 6.28*np.random.rand()-3.14]]).T), axis=1)
# print('Initial conditions:')
# print(initial_conditions)

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)
## Visualize goals, obstacles and the base
r.axes.add_patch(patches.Circle((0., 0.), 0.2, fill=True))          # Obstacle 1
r.axes.add_patch(patches.Circle((-0.5, 0.5), 0.2, fill=True))         # Obstacle 2
r.axes.add_patch(patches.Circle((-0.5, -.5), 0.2, fill=True))                # Obstacle
r.axes.add_patch(patches.Circle((1.4, 1.4), 0.2, fill=False, zorder=10))         # Base

#from http://www.sharejs.com
class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
args = DottableDict()
args.allowDotting() 
args.hazards_radius = 0.2
args.hazards_locations =  np.array([[0., 0.], [-0.5, 0.5], [-0.5, -.5]]) 
args.u_min = np.array([-1,-1])
args.u_max = np.array([1,1])

cbflayer = CascadeCBFLayer(args)
def is_done(x):
    """
    判断是否完成目标、被碰撞、智能体是否出界
    """
    self_state = x[0]
    other_states = x[1:]

    # Check boundaries
    if(self_state[1]>1.5 or self_state[1]<-1.5 or self_state[0]>1.5 or self_state[0]<-1.5):
        print('Out of boundaries !!')
        return True
    # Reached goal?
    if (1.35<=self_state[0]<=1.45 and 1.35<=self_state[1]<=1.45):
        print('Reach goal successfully!')
        return True

    for idx in range(np.size(other_states, 0)):
        # if(other_states[idx][0]>1.5 or other_states[idx][0]<-1.5 or other_states[idx][1]>1.5 or other_states[idx][1]<-1.5 ):
        #     print('Vehicle %d is out of boundaries !!' % idx+1)
        #     return True
        distSqr = (self_state[0]-other_states[idx][0])**2 + (self_state[1]-other_states[idx][1])**2
        if distSqr < (0.2)**2:
            print('Get caught, mission failed !')
            return True
    
    return False


x = r.get_poses().T
r.step()

i=0
times = 0


while (is_done(x)==False):
    # print('\n----------------------------------------------------------')
    # print("Iteration %d" % times)

    x = r.get_poses().T

    # Observe & Predict
    u_norm = Simple_Catcher(np.array([1.4,1.4]), x[0])

    enemy_x = np.delete(x[1:],2,1)
    # print(enemy_x)
    other_s = np.vstack((enemy_x,args.hazards_locations))
    u_safe =cbflayer.get_u_safe(u_norm,x[0],other_s)
    u_safe =u_norm + u_safe

    dxu = np.zeros([N,2])
    dxu[0] = np.array([u_safe[0],u_safe[1]])

    for idx in range(1, N):
        # defender_u = Simple_Catcher(x[0],x[idx])
        # dxu[idx] = defender_u
        # dxu[idx] = np.array([0, 0]) 
        dxu[idx] = np.array([0.15, 0.1]) 
    # for idx in range(3, N)
        # defender_u = Simple_Catcher(x[0],x[idx])
        # dxu[idx] = defender_u 

    r.set_velocities(np.arange(N), dxu.T)

    times+=1
    i+=1
    r.step()
    # print('----------------------------------------------------------\n')

r.call_at_scripts_end()
