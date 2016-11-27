import numpy as np
from itertools import product

from learning.model_free import Problem
from learning.model_free import qlearning
from pybrain.utilities import one_to_n
import pylab as pl
class BipedWalking(Problem):
    
     # the number of action values the environment accepts
    nstates = 162
    
    # the number of sensor values the environment produces
    nactions = 2
    
    time_step = .05
    
    #properties of the robot
    m1 = .5 #mass of shin
    m2 = 4  #mass of high
    l1 = .2 #length of shin
    l2 = .2 #length of thigh
    k1 = .01 #stiffness
    c1 = .001 #damping coefficient
    
    #constants
    g = 9.81
    theta1_bounds = np.array(
        [-np.inf, -.21, -.11, 0, 0.11, .21, np.inf])
    theta1d_bounds = np.array(
        [-np.inf,-2.9,2.9,np.inf])
    theta2_bounds = np.array(
        [-np.inf, -0.104, 0.104, np.inf])
    theta2d_bounds = np.array(
            [-np.inf, -1.05,  1.05, np.inf])
    nbins_across_dims = [
            len(theta1_bounds) - 1,
            len(theta1d_bounds) - 1,
            len(theta2_bounds) - 1,
            len(theta2d_bounds) -1]
    magic_array = np.cumprod([1] + nbins_across_dims)[:-1]
    
    def __init__(self):
        self.sample_initial_state()
        self.a = [10,-10]
        Problem.__init__(self,self.nstates,self.nactions)
        
    
    def sample_initial_state(self):
        """ Most environments will implement this optional method that allows for reinitialization. 
        """
        self.t = 0
        theta1 = 0
        theta1d = 0
        theta2 = 0
        theta2d = 0
        self.h = self.l1 + self.l2
        self.sensors = np.array([theta1,theta1d,theta2,theta2d])
        s = self.getObservation()
        return s
        
    def actions(self,s):
        return [0,1]
        
        
    def getBin(self, theta1, theta1d, theta2, theta2d):
      
    
        bin_indices = [
                np.digitize([theta1], self.theta1_bounds)[0] - 1,
                np.digitize([theta1d], self.theta1d_bounds)[0] - 1,
                np.digitize([theta2], self.theta2_bounds)[0] - 1,
                np.digitize([theta2d], self.theta2d_bounds)[0] - 1
                ]
        return np.dot(self.magic_array, bin_indices)

    def getBinIndices(self, linear_index):
        """Given a linear index (integer between 0 and outdim), returns the bin
        indices for each of the state dimensions.
        """
        return linear_index / self.magic_array % self.nbins_across_dims

    def getObservation(self):
        (theta1, theta1d, theta2, theta2d) = self.sensors
#        state = one_to_n(self.getBin(theta1, theta1d, theta2, theta2d),
#                self.nstates)

        return self.getBin(theta1, theta1d, theta2, theta2d)
        
    def state_reward(self, s, a):
        self.t += 1
        T = 10 - (20*a)
        #print(T, " torque ")
        self.step(T)
        r = self.getReward()
        return self.getObservation(),r
        
        
    def getReward(self):
        if self.h < .38:
            return -1.0
        elif self.t == 5000:
            return 10.0
        else:
            return 0.0
        
    def step(self,T):
        [theta1,theta1d,theta2,theta2d] = self.sensors
        self.h = self.l1*np.cos(theta1) + self.l2*np.cos(theta2)
        
        c11 = (self.m1 + self.m2)*(self.l1**2)
        c12 = self.m2*self.l2*self.l1*np.cos(theta1 - theta2)
        c21 = self.m2*self.l1*self.l2*np.cos(theta1-theta2)
        c22 = self.m2*self.l2
        l11 = self.m2*self.l1*self.l2*(theta2d**2)*np.sin(theta1-theta2) - self.g*self.l1*(self.m1+self.m2)*np.sin(theta1) -T +(self.k1*theta1)+(self.c1*theta1d)
        l22 = -self.m2*self.l1*self.l2*(theta1d**2)*np.sin(theta1-theta2)-self.m2*self.l2*self.g*np.sin(theta2)
        
        theta1d += ((c12*l22-c22*l11)/(c11*c22-c12*c21))*self.time_step
        theta1 += theta1d*self.time_step
        theta2d += ((c21*l11-l22*c11)/(c11*c22-c12*c21))*self.time_step
        theta2 += theta2d*self.time_step
        

        theta1 = np.clip(theta1,-0.21,0.21)
        theta2 = np.clip(theta2,-0.104,0.104)
        theta1d = np.clip(theta1d,-2.09,2.09)
        theta2d = np.clip(theta2d,-1.05,1.05)
        
        self.sensors = np.array([theta1,theta1d,theta2,theta2d])
        print(self.sensors, " height " , str(self.h))
    def is_final(self, s):
        if self.h < .38:
            return True
        elif self.t == 5000:
            return True
        else:
            return False
        
    
def main():
    problem = BipedWalking()
    Q,pi, v = qlearning(problem, 100, epsilon=0.1, alpha=0.5, gamma=0.5)

        
if __name__ == "__main__":
    main()        
