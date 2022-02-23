
import multiprocessing as mp
import time
from RoboticArm import RoboticArm
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation.state_machine import States, UnitySensingStateMachine
from project.simulation.controller import Control
from project.simulation.utilities import *
from project.simulation.mjremote import mjremote


def run(with_unity=True, sim_params=None, sim_positions=None):
   
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
    sceneChangeCounter = 0
    if not unity:
        while True:
            # moore.eval()
            control.PID()
            scene.show_step()
            print(robot.get_joints_pos())
    arm = RoboticArm
    while True:
        if sceneChangeCounter != sim_params[10]:
            pos = [sim_params[5]/100, sim_params[6]/100, sim_params[7]/100]
            moore.set_external_target(pos)
            moore.curr_state = States.INIT
            sceneChangeCounter = sim_params[10]

        if sim_params[2] == 1:
            moore.eval()
            # try while sceneChangeCounter = sim_params[10] : pass
        else:
            control.theta_d = robot.get_joints_pos()
            
        if sceneChangeCounter != sim_params[10]:
            pos = [sim_params[5]/100, sim_params[6]/100, sim_params[7]/100]
            moore.set_external_target(pos)
            moore.curr_state = States.INIT
            sceneChangeCounter = sim_params[10]

        control.PID(speed=sim_params[4])
        scene.show_step()
        if sim_positions is not None:
            pos = robot.get_joints_pos()
            for i, p in enumerate(pos):
                sim_positions[i] = p
        

def activate_arm(sim_positions):
        robotic_arm = RoboticArm()
        nap_configuration = [-0.5*np.pi, -0.6*np.pi, 1*np.pi, 0.5*np.pi, 0.4*np.pi, 0]
        robotic_arm.enable_torque()
        robotic_arm.set_map_from_nap(nap_configuration)
        while True:
            if sim_positions[0] != 0:
                robotic_arm.set_position_from_sim(sim_positions)
            

if __name__ == '__main__':
    run(False)
