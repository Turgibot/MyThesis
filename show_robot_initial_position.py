from os import stat
from sys import set_asyncgen_hooks
from zipfile import Path
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation.state_machine import StateMachine
from project.simulation.controller import Control
from project.simulation.utilities import *
def main():
    xml_path = "./project/models/vx300s/vx300s.xml"
    scene = Mujocoation(xml_path)
    robot = Robot(scene.model, scene.simulation)
    control = Control(robot, scene.simulation)
    moore = StateMachine(robot, scene, control)
    while True:
        moore.eval()
        control.PID()
        scene.show_step()
        # print(robot.get_ee_position())
        
if __name__== "__main__":

    main()