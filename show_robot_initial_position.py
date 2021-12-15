from zipfile import Path
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation import path_maker as pm
from project.simulation.controller import Control

def main():
    xml_path = "./project/models/vx300s/vx300s.xml"
    scene = Mujocoation(xml_path)
    robot = Robot(scene.model, scene.simulation)
    controller = Control(robot, scene.simulation)
    thetas = np.array([0, 0, 0, 0, np.pi/2, 0])
    scene.simulation.data.qpos[:] = thetas
    scene.simulation.forward()
    
    print(robot.get_links_positions())
    scene.advance_once()

if __name__== "__main__":
    main()