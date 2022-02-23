
import multiprocessing as mp
import time
from RoboticArm import RoboticArm
from project.simulation.robot import Robot
from project.simulation.scene import *
from project.simulation.state_machine import SimpleStateMachine
from project.simulation.controller import Control
from project.simulation.utilities import *
from project.simulation.mjremote import mjremote


def run(with_unity):
   
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
    moore = SimpleStateMachine(robot, scene, control, orientation=1)
    # face_down_configuration = [-0.5*np.pi, 0,          0.5*np.pi, 0.5*np.pi, 0,          0]
    # control.theta_d =         [-0.5*np.pi, -0.6*np.pi, 1.0*np.pi, 0.5*np.pi, 0.4*np.pi, 0]
    arm = RoboticArm()
    arm.enable_torque()
    real_nap_config = arm.get_positions_in_rad()
    print(real_nap_config)
    robot.set_map_from_nap(real_nap_config)
    print(robot.map_sim_to_real(robot.nap))
    robot.take_a_nap()
    arm.set_position(robot.map_sim_to_real())
    while True:
        moore.eval()
        control.PID()
        scene.show_step()
        # print(robot.get_joints_pos())
        arm.set_position(robot.map_sim_to_real())

    
if __name__ == '__main__':
    run(False)
