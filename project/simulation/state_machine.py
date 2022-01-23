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

from .utilities import *
import mujoco_py as mjc

class StateMachine:
    def __init__(self, robot, scene, control, targets_pos=None):
        
        self.robot = robot
        self.scene = scene
        self.control = control
        self.simulation = self.scene.simulation
        self.model = self.scene.model
        if targets_pos:
            self.targets_pos = targets_pos
        else:
            self.targets_pos = [self.scene.get_target_pos_euler()[0], [-0.4, 0.1, 0.3],[0.4, 0.15, 0.55],[-0.17, 0.1, 0.1]]
        self.states = {}
        self.state = 'home'
        self.counter = 0
        self.curr_target_p = None 
        self.thetas = None
        self.home_th = 0.02
        self.target_th = 0.012
        self.th = 0.1
        self.first_run = True
        self.get_states()

    def get_states(self):
        self.states['home'] = [self.robot.home, self.robot.ee_home]
        self.states['right'] = [self.robot.right, self.robot.ee_right]
        self.states['bottom_right'] = [self.robot.bottom_right, self.robot.ee_bottom_right]
        self.states['left'] = [self.robot.left, self.robot.ee_left]
        self.states['bottom_left'] = [self.robot.bottom_left, self.robot.ee_bottom_left]
        self.states['target'] = [self.thetas, self.curr_target_p]
    
    def output(self):
        # reset
        if self.state is 'home':
            self.thetas = None
            self.curr_target_p = self.targets_pos[self.counter]
            
        elif self.state is 'target':
            if self.thetas is None:
                self.T_target = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
                self.T_target = np.r_[np.c_[self.T_target, self.curr_target_p], [[0,0,0,1]]]
                self.thetas = self.control.IK(self.T_target, sections=6)
                self.states['target'] = [self.thetas, self.curr_target_p]
        
        self.control.theta_d =  self.states[self.state][0]
        
    def next_state(self):
       
        dist = np.linalg.norm(self.robot.get_ee_position() - self.states[self.state][1])
        dist_to_target = np.linalg.norm(self.robot.get_ee_position() - self.curr_target_p)
        if self.state is 'home' and (dist < self.home_th or not self.first_run):
            if self.curr_target_p[0] >= 0.1:
                if self.curr_target_p[2] <= 0.3:
                    self.state = 'bottom_right'
                else:
                    self.state = 'right'
            elif self.curr_target_p[0] <= -0.1:
                if self.curr_target_p[2] <= 0.3:
                    self.state = 'bottom_left'
                else:
                    self.state = 'left'
            self.first_run = False
                    
        elif self.state is not 'target' and self.state is not 'home' and (dist < self.th or dist_to_target< self.th):
            self.state = 'target'
        elif dist < self.target_th:
            self.state = 'home'
            self.counter += 1
            self.counter %= len(self.targets_pos)
            self.simulation.data.set_mocap_pos("target",  self.targets_pos[self.counter])
        # print(dist)
    
    def eval(self):
        # print(self.state)
        self.output()
        self.next_state()

