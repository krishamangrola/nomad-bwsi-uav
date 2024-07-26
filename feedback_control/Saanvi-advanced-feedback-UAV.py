from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

_HEIGHT = 1.0
_VEL_CONST = 1.0
_TIME_STEP = 0.1
_THETA_LIMIT = np.pi/4.0

class PitchSlideCamera():
    '''Object that defines the dynamics of the simple slide-camera'''
    
    def __init__(self, x_0, v_0, theta_0, x_d, gamma_d=0.0, h=_HEIGHT, k=_VEL_CONST, theta_limit=_THETA_LIMIT):
        
        # state variables (hidden)
        self.__x = x_0
        self.__v = v_0
        
        # reference position (hidden)
        self.__x_d = x_d
        
        # reference angle (observed)
        self.gamma_d = gamma_d
        
        # parameters
        self.__h = h
        self.__k = k
        self.__theta_limit = theta_limit
        
        # control variables (observed, commanded)
        self.__theta = theta_0
        
    def get_theta(self):
        return self.__theta
        
    def sense_gamma(self):
        # calculate angle from camera center line to target
        return  np.arctan2(self.__x - self.__x_d, self.__h) - self.__theta
    
    def _get_hidden_position(self):
        
        return self.__x
    
    def _get_hidden_position_desired(self):
        return self.__x_d
    
    def _get_hidden_velocity(self):
        return self.__v
    
    def actuate_theta_command(self, theta_cmd, dt=_TIME_STEP):
        self.__theta = min(self.__theta_limit, max(theta_cmd, -self.__theta_limit))
        self.__v += self.__k*np.sin(self.__theta)*dt
        self.__x += self.__v*dt 

def p_control(y_err, kp):
    ''' compute the actuator command based on proportional error between output and desired output
    Args:
    y_err: y_des - y where y is the output variable of the plant
    '''
    
    # TODO: write a proportional control law (hint: it is a single line, very simple equations)
    cmd = kp * y_err
    
    return cmd

def pd_control(y_err, y_err_prev, dt, kp, kd):
    '''compute the actuator command based on proportional and derivative error between output and target
    Args:
    y_err: y_des - y where y is the output variable of the plant
    y_err_prev: previous step y_des - y
    '''
    
    # TODO: write a proportional+derivative control law
    cmd = (kp * y_err) + (((y_err - y_err_prev) / dt) * kd)
    
    return cmd

assert np.isclose(pd_control(0.0, 1.0, 0.1, 1.0, 1.0), -10.0)

def custom_control():
    '''custom-made controller, if you want to develop one
    Args:
    '''
    pass

# Control gains
# YOUR CODE HERE
kp = 0.6
kd = 0.092


# Control inputs
dt = _TIME_STEP
t_final = 50.0

# intial conditions (position, velocity and targe position)
x_0 = 0.0
v_0 = 0.0
theta_0 = 0.0
x_des = 1.0

# create SimpleSlideCamera with initial conditions
pscam = PitchSlideCamera(x_0, v_0, theta_0, x_des)

# initialize data storage
data = dict()
data['t'] = []
data['theta_cmd'] = []
data['theta'] = []
data['err_gamma'] = []
data['x_hidden'] = []
data['v_hidden'] = []
t = 0.0
err_gamma_prev = 0.0
while t < t_final:
    t += dt
    
    # SENSOR: sense output variable gamma (angle from camera centerline to target) and calculate error from desired
    err_gamma = pscam.gamma_d - pscam.sense_gamma()
    
    # CONTROLLER: call theta control algoritm
    theta_cmd = pd_control(err_gamma, err_gamma_prev, dt, kp, kd)
    
    # ACTUATOR: send velocity command to plant
    pscam.actuate_theta_command(theta_cmd)
    
    # store data
    err_gamma_prev = err_gamma
    data['t'].append(t)
    data['theta_cmd'].append(theta_cmd)
    data['theta'].append(pscam.get_theta())
    data['err_gamma'].append(err_gamma)
    data['x_hidden'].append(pscam._get_hidden_position())
    data['v_hidden'].append(pscam._get_hidden_velocity())
    
# Plot Data
handle_position, = plt.plot(data['t'], data['x_hidden'], label='position (hidden)[m]')
handle_velocity, = plt.plot(data['t'], data['v_hidden'], label='velocity (hidden)[m/s]')
handle_err_gamma, = plt.plot(data['t'], data['err_gamma'], label='gamma error [rad]')
handle_theta, = plt.plot(data['t'], data['theta'], label='theta [rad]')
plt.legend(handles=[handle_position, handle_velocity, handle_err_gamma, handle_theta])
plt.show()

# Questions

# Q1. Can you design a controller that is capable of converging the gamma error to zero?

# Yes, by adding an integral term to the PD control law, it would maybe reduce the oscillation.

# Q2. The time scale to make this control converge is on the order of 10s of seconds. Our real drones, which are much more complex control problems, are capable of converging on a target much faster with far less oscillation. Can you give explanations why our drone controllers perform so much better than this controller?

# Our drones are able to converge to the target in shorter amounts of time because they use a PID controller, which combines Kp, Ki, and Kd to combat steady-state error and any overshoots in a more percise manner.