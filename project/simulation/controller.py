"""
This class implements a PID that calculates the 
position error relative to the target and outputs a vector of correcting torques
"""

import numpy as np



class PID:
    
    def __init__(self, data, **kwarg):
        
        self.data = data
        self.params =  {'Kv': 20, 'Kp': 200, 'Ko': 200, 'Ki': 0, 'vmax': [0.5, 0]}              
        self.kp = 0.1
        self.ki = 0.1
        self.kd = 0.1
        self.integral = 0
        self.proportional = 0
        self.derivative = 0
        self.previous_err = self.get_error() 
    
    def get_error(self):
        target = self.data.get_target() 
        ee = self.data.get_ee_position()
        return ee-target
    
    def integrate(self):
        self.integral += self.ki*self.get_error()
    
    def proportionate(self):
        self.proportional = self.pi*self.get_error()
    
    def derivate(self):
        curr_error = self.get_error()
        self.derivative =  curr_error-self.previous_err
        self.previous_err = curr_error
    
