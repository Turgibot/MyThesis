from zipfile import Path
from project.simulation import controller
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation import path_maker as pm
from project.simulation.controller import Control
from project.simulation.utilities import *
def main():
    xml_path = "./project/models/vx300s/vx300s.xml"
    scene = Mujocoation(xml_path)
    robot = Robot(scene.model, scene.simulation)
    control = Control(robot, scene.simulation, theta_d=robot.home)

    while True:
        control.PID()
        scene.show_step()
        print(robot.get_ee_position())
    
    # print(control.FK())
    

if __name__== "__main__":

    main()