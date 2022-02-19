
import multiprocessing as mp
import time
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation.state_machine import States, UnitySensingStateMachine
from project.simulation.controller import Control
from project.simulation.utilities import *
from project.simulation.mjremote import mjremote


def run(with_unity=True, sim_params=None):
   
    unity = None
    if with_unity:
        time.sleep(2)
        unity = mjremote()
        while not unity._s:  
            unity.connect() 
            print("conecting...")
        print("SUCCESS")
    
    xml_path = "./project/models/vx300s/vx300s_face_down.xml"
    scene = Mujocoation(xml_path, unity)
    robot = Robot(scene.model, scene.simulation)
    control = Control(robot, scene.simulation)
    moore = UnitySensingStateMachine(robot, scene, control, orientation=1)
    set_once = False
    while True:
        if sim_params[2] == 1:
            if set_once:
                pos = [sim_params[5]/100, sim_params[6]/100, sim_params[7]/100]
                moore.set_external_target(pos)
                moore.curr_state = States.INIT
                set_once = False
            moore.eval()
        else:
            control.theta_d = robot.get_joints_pos()
            set_once = True
        control.PID(speed=sim_params[4])
        scene.show_step()
        
    
if __name__ == '__main__':
    run(False)
