
import multiprocessing as mp

from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation.state_machine import StateMachine
from project.simulation.controller import Control
from project.simulation.utilities import *
from project.simulation.mjremote import mjremote


def run(with_unity):
   
    unity = None
    if with_unity:
        unity = mjremote()
        while not unity._s:
            unity.connect()
            print("conecting...")
        print("SUCCESS")

    xml_path = "./project/models/vx300s/vx300s.xml"
    scene = Mujocoation(xml_path, unity)
    robot = Robot(scene.model, scene.simulation)
    control = Control(robot, scene.simulation)
    moore = StateMachine(robot, scene, control)
    while True:
        moore.eval()
        control.PID()
        scene.show_step()

    