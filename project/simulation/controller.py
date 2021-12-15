"""
This class implements a PID that calculates the 
torque needed by an input of the desired joint angles
"""

"""
Control is achieved by applying torque
Real robot has a builtin PID controller so the input is simply the joint values
"""


import numpy as np
from .utilities import *

class Control:
    def __init__(self, robot, simulation) -> None:
        self.robot = robot
        self.model = robot.model
        self.simulation = simulation

        self.FK(thetalist= np.array([0, 0, 0, 0, np.pi/2, 0])
)

# -----------------------------------------------------------------------------
# FORWARD KINEMATICS
# -----------------------------------------------------------------------------
    def FK(self, thetalist):
        print(FKinSpace(self.robot.M, self.robot.Slist, thetalist))

    