"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class solves the IK problem and creates a list of states.
Each state is a dictionary of joint torques

"""
import random
import numpy as np
from sympy import N

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

class States:
    INIT = 0
    HOME = 100
    STEPS = 200
    RETURN = 3000
    
class SimpleStateMachine:
    def __init__(self, robot, scene, control, targets_pos=None) -> None:
        self.robot = robot
        self.scene = scene
        self.control = control
        self.simulation = self.scene.simulation
        self.model = self.scene.model
        
        # targets are [x, y, z] coordinates
        self.targets = [self.scene.get_target_pos_euler()[0]]
        #th to position ee infront of target
        self.y_th = 0.5
        self.thetas = None
        self.curr_state_target = self.robot.ee_home
        self.curr_final_target = self.targets[0]
        self.target_orientation = np.array([[0, 0, 1],
                                            [1, 0, 0],
                                            [0, 1, 0]])
        self.curr_state_configuration = []
        self.steps_positions = []
        self.steps_thetas = []
        self.reached_th = 0.02
        self.curr_state = States.INIT
        self.prev_state = States.INIT
        self.targets_counter = 0
        self.steps_counter = 0
        self.num_steps = 100
        self.distance = 0
        self.is_return = True
        self.get_random_targets()
    
    def get_random_targets(self):
        self.curr_final_target[2] = self.curr_final_target[2]+0.05
        y = self.curr_final_target[1]
        z = self.curr_final_target[2]
        num = 10
        for i in range(num):
            x = random.randrange(-30, 30, num)/100
            if abs(x) >0.1:
                self.targets.append([x,y,z])


    def next_state(self):

        # next state depending on current state and the distance between the EE to the current state target

        self.distance = np.linalg.norm(self.robot.get_ee_position() - self.curr_state_target)
        
        # init state to get data for the current target
        if self.curr_state == States.INIT:
            #set final target
            self.curr_final_target = self.targets[self.targets_counter]
            self.simulation.data.set_mocap_pos("target",  self.curr_final_target)
            #set state target and configuration
            self.curr_state_target = self.robot.ee_home
            self.curr_state_configuration = self.control.FK(self.robot.home)
            #next state
            self.curr_state = States.HOME
            self.steps_counter = 0
            #next time that the INIT state is reached move on to the next target
            self.targets_counter += 1
            self.targets_counter %= len(self.targets)
        
        # primary condition to move to the next step is the distance to the current state target
        elif self.distance < self.reached_th:
            if self.curr_state == States.HOME:
                # create the next states, positions
                self.steps_positions = [] #empty list
                curr_ee_pos = self.robot.get_ee_position()
                diff = self.curr_final_target - curr_ee_pos
                for i in range(self.num_steps):
                    self.steps_positions.append((curr_ee_pos+((i+1)/self.num_steps)*diff))
                # set the next state as the steps
                self.curr_state = States.STEPS 
                self.set_step()
            
            elif self.curr_state == States.STEPS+self.steps_counter:
                self.steps_counter += 1
                if self.steps_counter >= self.num_steps:
                    if self.is_return:
                        self.curr_state = States.RETURN
                    else:
                        self.curr_state = States.INIT
                    self.steps_counter = 0
                    self.is_return = not self.is_return
                else:
                    self.set_step()

            elif self.curr_state == States.RETURN:
                self.steps_positions = [] #empty list
                curr_ee_pos = self.robot.get_ee_position()
                diff = self.robot.ee_home - curr_ee_pos 
                for i in range(self.num_steps):
                    self.steps_positions.append((curr_ee_pos+((i+1)/self.num_steps)*diff))
                # set the next state as the steps
                self.curr_state = States.STEPS 
                self.set_step()
                
                # self.steps_counter -= 1
                # if self.steps_counter <= int(self.num_steps*0.03):
                #     self.curr_state = States.INIT
                # else:
                #     self.step_back()


    def output(self):
        #output is dependant of the current state
        if self.curr_state != self.prev_state:
            if self.curr_state == States.HOME or self.curr_state == States.INIT:
                self.control.phase = 0
                self.control.theta_d = self.robot.home
            else:
                self.control.phase = 1
                thetas = self.control.IK(self.curr_state_configuration)
                self.steps_thetas.append(thetas)    
                self.control.theta_d = thetas
            
            if self.steps_counter >= int(self.num_steps*0.93):
                self.control.phase = 2
            
            self.prev_state = self.curr_state
        
    def set_step(self):
        self.curr_state_target = self.steps_positions[self.steps_counter]
        self.curr_state_configuration = np.r_[np.c_[self.target_orientation, self.curr_state_target], [[0,0,0,1]]]
        self.curr_state = States.STEPS + self.steps_counter
    
    def step_back(self):
        self.curr_state_target = self.steps_positions[self.steps_counter]
        self.curr_state = States.RETURN + self.steps_counter

    def eval(self):
        self.next_state()
        self.output()