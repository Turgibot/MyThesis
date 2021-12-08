"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class solves the IK problem and creates a list of states.
Each state is a dictionary of joint torques

"""
import numpy as np

from .robot import Robot
from .utilities import *
import mujoco_py as mjc
from .controller import PID

class Path:
    def __init__(self, scene):
        
        self.data = Robot(scene)
        self.model = self.data.model
        self.simulation = self.data.simulation
        self.controller = PID(self.data)
       

    # def calculate_states(self):
    #     for target in self.data.target_coor:
    
    # This func moves the arm to the desired configuration artificially in a single step.
    def go_home(self):
        for i in self.home:
            self.simulation.data.qpos[self.model.joint_dict[i]['position_address']] = self.home[i]
        self.simulation.forward()