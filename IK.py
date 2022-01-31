import os, time
import multiprocessing as mp
from zipfile import Path
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation.state_machine import StateMachine
from project.simulation.controller import Control
from project.simulation.utilities import *
from project.simulation.mjremote import mjremote

with_unity = False

def main():
   
    unity = None
    if with_unity:
        time.sleep(5)
        unity = mjremote()
        while not unity._s:
            unity.connect()
            print("conecting...")
        print("SUCCESS")

    xml_path = "./project/models/vx300s/vx300s_shelves.xml"
    scene = Mujocoation(xml_path, unity)
    robot = Robot(scene.model, scene.simulation)
    control = Control(robot, scene.simulation)
    targets_pos = [[0, 0.3, 0.34], [0, 0, 0]]
    robot.simulation.data.set_mocap_pos("target",  targets_pos[0])
    moore = StateMachine(robot, scene, control, targets_pos)
    while True:
        moore.eval()
        control.PID()
        scene.show_step()
    
def start_unity():
    if with_unity:
        print("unity is loading")
        os.system("./Robot/unitybot.x86_64 &")
if __name__== "__main__":
    unity = mp.Process(target=start_unity)
    main = mp.Process(target=main)
    unity.start()
    main.start()
    unity.join()
    main.join()