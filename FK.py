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

    xml_path = "./project/models/vx300s/vx300s.xml"
    scene = Mujocoation(xml_path, unity)
    robot = Robot(scene.model, scene.simulation)
    control = Control(robot, scene.simulation)

    thetas = np.array([np.pi/2, -np.pi/2, 0, np.pi/2, np.pi/2, np.pi/2])
    
    control.theta_d = thetas
    
    while True:
        control.PID()
        scene.show_step()
        # print(robot.get_ee_position())
    
    unity.close()    
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