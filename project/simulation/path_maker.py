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

class Path:
    def __init__(self, scene):
        
        self.robot = Robot(scene)
        self.model = self.robot.model
        self.simulation = self.robot.simulation
       

    # def calculate_states(self):
    #     for target in self.data.target_coor:
    
    # This func moves the arm to the desired configuration artificially in a single step.
    def go_home(self):
        for i in self.robot.home:
            self.simulation.data.qpos[self.model.joint_dict[i]['position_address']] = self.robot.home[i]
        self.simulation.forward()