"""
This class implements a PID that calculates the 
torque needed by an input of the desired joint angles
"""

"""
Control is achieved by applying torque
Real robot has a builtin PID controller so the input is simply the joint values
"""


import re
import numpy as np
from .utilities import *
from .kinematics import Kinematics
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

from project.simulation import kinematics
class Control:
    def __init__(self, robot, simulation, theta_d=None) -> None:
        self.robot = robot
        self.model = robot.model
        self.simulation = simulation
        self.kinematics = Kinematics(robot)
        self.theta_d = theta_d if theta_d is not None else np.array(self.robot.home)
        self.thetalist= np.array(self.theta_d)
        self.d = np.zeros(self.robot.n_joints)
        self.i = np.zeros(self.robot.n_joints)
        self.prev_err = np.subtract(self.theta_d, self.simulation.data.qpos[:])
        self.kp = 0.25
        self.ki = 0.00000001
        self.kd = 0.1
        
        
# -----------------------------------------------------------------------------
# FORWARD KINEMATICS
# -----------------------------------------------------------------------------
    
    def FK(self, thetalist=None):
        if thetalist is None:
            thetalist = self.theta_d
        fk = self.kinematics.FK(self.robot.M, self.robot.s_poe, thetalist)
        return fk

   
    #calculate the necessary velocity to drive each joint to a desired theta_d 
    def PID(self):
               
        err = np.subtract(self.theta_d, self.simulation.data.qpos[:])
        self.i = np.add(self.i, err)
        self.d = np.subtract(err,  self.prev_err)
        self.prev_err = np.copy(err)
        v = self.kp*err + self.ki*self.i + self.d*self.d
        u = -self.get_gravity_bias()[:]
        self.simulation.data.ctrl[:] = u
        self.simulation.data.qvel[:] = v

    def get_gravity_bias(self):       
        """ Returns the effects of Coriolis, centrifugal, and gravitational forces """
        g = -1 * self.simulation.data.qfrc_bias[:]
        return g

    def IK(self, T_target, sections=5):
        eomg = 0.000000000000001
        ev = 0.00000000000001
        thetas = self.kinematics.trajectoryIK(T_target, eomg, ev, sections)
        # for i in range(len(thetas)):
        #     if thetas[i]>np.pi:
        #         thetas[i]-=2*np.pi
        #     elif thetas[i]<-np.pi:
        #         thetas[i]+=2*np.pi
        return thetas

        

