# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace
# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation

import os
import pygame

class PS4Controller():
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    def __init__(self):
        """Initialize the joystick components"""
        
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        self.buttons = {
                    0: {'key':'Cross',    'value': False},
                    1: {'key':"Circle",   'value': False},
                    2: {'key':"Square",   'value': False},
                    3: {'key':"Triangle", 'value': False},
                    4: {'key':"Share",    'value': False},
                    5: {'key':"P",        'value': False},
                    6: {'key':"Options",  'value': False},
                    7: False,
                    8: False,
                    9:  {'key':"L1",      'value': False},
                    10: {'key':"R1",      'value': False},
                    11: {'key':"Up",      'value': False},
                    12: {'key':"Down",    'value': False},
                    13: {'key':"Left",    'value': False},
                    14: {'key':"Right",   'value': False},
                    15: {'key':"Keypad",  'value': False}}

        self.axis = {
                    0: {'key': 'Left_horizontal',  'value': 0.0},
                    1: {'key': 'Left_vertical',    'value': 0.0},
                    2: {'key': 'Right_horizontal', 'value': 0.0},
                    3: {'key': 'Right_vertical',   'value': 0.0},
                    4: {'key': 'L2', 'value': -1.0},
                    5: {'key': 'R2', 'value': -1.0}

        }

    def listen(self, robot_state, arm, actuation_function):

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self.axis[event.axis]['value'] = round(event.value, 2)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.buttons[event.button]['value'] = True
                elif event.type == pygame.JOYBUTTONUP:
                    self.buttons[event.button]['value'] = False

                if actuation_function is not None:
                    actuation_function(robot_state, self.axis, self.buttons, arm)
