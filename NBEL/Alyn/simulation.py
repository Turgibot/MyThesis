"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphic Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

This work is based on ABR's adaptive controller availible at: 
https://github.com/abr/abr_control/tree/master/abr_control 
Using this code is subjected to ABR's licensing

Adaptive control theory is based on:
DeWolf, Travis, Terrence C. Stewart, Jean-Jacques Slotine, and Chris Eliasmith. 
"A spiking neural model of adaptive arm control." 
Proceedings of the Royal Society B: Biological Sciences 283, no. 1843 (2016): 20162134.

Physical simulation is based the MuJoCo simulator (http://www.mujoco.org)
Using the simulator is subject to acquiring a license for MuJoCo (https://www.roboti.us/license.html)

Adaptive control is implemented with the nengo framework (nengo.ai)

Operational space controller is based on:
Khatib, Oussama. 
"A unified approach for motion and force control of robot manipulators: The operational space formulation." 
IEEE Journal on Robotics and Automation 3.1 (1987): 43-53. 
"""

import mujoco_py as mjc
import numpy as np
import glfw
import time
from datetime import datetime
import logging

from OSC import OSC
from utilities import euler_from_quaternion
from adaptive_control import DynamicsAdaptation  

class Controller:
    
    def __init__(self, model):
        
        # Intializing Operational Space Controller (OSC)         
        self.controller = OSC(model)
    
    def generate(self, position, velocity, target):
        return self.controller.generate(position, velocity, target)
    
        
class Simulation:
    
    def __init__(self, base_dir,             # Directory. Used to initiate a logger
                 model,                      # Instance of the mechanical model of the arm   
                 controller,
                 init_angles      = None,    # Initial configuration of the arm null position.
                 target           = None,    # Array of target coordinates in reference to the EE position
                 return_to_null   = False,   # Return to home position before a approaching a new target
                 th               = 2e-2,    # Treshold for successful approach to target
                 sim_dt           = 0.01,    # Simulation time step
                 external_force   = None,    # External force field (implemented with scaled gravity)
                 adapt            = False,   # Using adaptive controller
                 n_gripper_joints = 0,       # Number of actuated gripping points    
                ):      

        self.model                = model
        self.target               = target
        self.return_to_null       = return_to_null
        self.th                   = th
        self.dt                   = sim_dt
        self.external_force_field = external_force
        self.n_gripper_joints     = n_gripper_joints
        self.controller           = controller
        
        if init_angles is None:
            self.init_angles      = model.ref_angles
          
        #Initialize a logger (debug, info, warning, error, critical)
        self.logger = logging.getLogger(__name__)  
        self.logger.setLevel(logging.DEBUG)
        log_name = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
        file_handler = logging.FileHandler(base_dir +'log/{}.log'.format(log_name))
        formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info('Logger initialized')
        
        # Initiating the monitor dictionary
        if target is not None:
            self.monitor_dict = {}
            for i, t in enumerate(target):
                self.monitor_dict[i] = {'error': [],       # Delta between current to target location
                                        'ee': [],          # Position of the end-effector
                                        'q': [],           # Joint angles
                                        'dq': [],          # Joint velocity
                                        'steps' : 0,       # number of steps to mitigate the target
                                        'target': t,       # Target coordinates (referenced to the arm null position)
                                        'target_real': 0}  # Target coordinates (referenced to the world)
                self.logger.info('Target {}: {}'.format(i, t))
        
        # Get number of joints
        self.n_joints     = int(len(
            self.model.simulation.data.get_body_jacp('EE')) / 3) # Jacobian translational component (jacp)
        self.logger.info('Number of joints: {}'.format(self.n_joints))
        
        # Initiating pose
        self.model.goto_null_position()
        self.null_position = self.model.get_ee_position()
        self.logger.info('Null position: {}'.format(self.null_position))
        
        # Initialize adaptive controller
        self.adaptation = adapt
        if adapt:          
            self.adapt_controller = DynamicsAdaptation(
                n_input           = 10,     # Applying adaptation to the first 5 joints, having 5 angles and 5 velocities
                n_output          = 5,      # Retrieving 5 adaptive signals for the first 5 joints
                n_neurons         = 5000,   # Number of neurons for neurons per ensemble
                n_ensembles       = 5,      # Defining an ensemble for each retrived adaptive signals                             
                pes_learning_rate = 1e-4,   # Learning rate for the PES online learning rule
                means             = [       # Scaling the input signals with means / variances of expected values. 
                                     0.12,  2.14,  1.87,  4.32, 0.59, 
                                     0.12, -0.38, -0.42, -0.29, 0.36],
                variances         = [
                                     0.08, 0.6, 0.7, 0.3, 0.6, 
                                     0.08, 1.4, 1.6, 0.7, 1.2]
            )
            self.logger.info('Adaptive controller initialized')
            
    def visualize(self):
        """ visualizing the model with the initial configuration of the arm """
        
        self.model.visualize()
    
    def simulate(self, 
                 steps = None): # Number of maximum allowable steps for mitigating the target
        """ Simulating the model """
        
        if self.target is None:
            print('A target eas not defined. Try to visualize instead.')
            return
         
        # Signifying termination of the simulation 
        breaked = False
            
        # Iterate over the predefined targets ---------------------------------------------------
        for exp in self.monitor_dict:
            
            self.logger.info('Moving to target {} at: {}'.format(exp, self.monitor_dict[exp]['target']))
            
            # Terminate the simulation if it was signaled to using the ESC key
            if breaked:
                break
            
            # Retrieving the position of the target in world's coordinates
            target = self.null_position + self.monitor_dict[exp]['target']
            self.monitor_dict[exp]['target_real'] = np.copy(target[:3])
            self.logger.info('Target position in world coordinates: {}'.format(exp, self.monitor_dict[exp]['target_real']))
            
            # Setting the location of target (sphere; defined in the XML model) in the simulation
            self.model.simulation.data.set_mocap_pos("target", target)

            # Keeping track of the simulation's number of steps
            step = 0 
            
            # Initializing error
            error = float("inf")

            while True: # Execute simulation -----------------------------------------------------

                # Breaking conditions ------------------------------------------------------------
                
                # Keeping track of the steps and moving to the next target if exceeding limit
                step += 1       
                if steps is not None:
                    if step > steps:
                        self.monitor_dict[exp]['steps'] = step
                        self.logger.critical(
                            'Simulation terminated by maxed number of steps (@ {}); Moving to next target'.format(step))
                        break

                # Terminate simulation with ESC key
                if self.model.viewer.exit:
                    breaked = True
                    self.logger.warning('Simulation manually terminated @ step {}'.format(step))
                    break
                    
                # Terminate, or move to the next target when the EE is within 
                # the threshold value of the target
                if error < 1e-2:
                    self.monitor_dict[exp]['steps'] = step
                    if self.return_to_null:
                        self.goto_null_position()
                    self.logger.info(
                        'Destination reached @ step {} with error {}; Moving to next target'.format(step, error))
                    break 

                # Calculating control signals ----------------------------------------------------
                
                # Force array which will be sent to the arm
                u = np.zeros(self.model.n_joints)
                
                # Retrieve the joint angle values and velocities
                position, velocity = (self.model.get_angles(), self.model.get_velocity())

                # Request the OSC to generate control signals to actuate the arm
                u = self.controller.generate(position, velocity, target)

                # Converting the retrieved dictionary to force arrays to be send to the arm
                # Only the first 5 actuators are activated. 
                # The six'th actuator controls EE orientation.
                position_array = [np.copy(position[i]) for i in range(5)]
                velocity_array = [np.copy(velocity[i]) for i in range(5)]
                
                # If adaptation mode is on, that retireve the adapt signals
                if self.adaptation:
                    u_adapt = np.zeros(self.model.n_joints)
                    # Retrieveing the adapt signals from the adaptive controller
                    u_adapt[:5] = self.adapt_controller.generate(
                        # 10 inputs constituting the arm's joints' angles and velocities
                        input_signal    = np.hstack((position_array, velocity_array)),
                        # Training signal for the controller. 
                        # Training signal is the actuation values retrieved before, 
                        # without the gravitational force field. 
                        training_signal = np.array(self.controller.controller.training_signal[:5]),
                    )
                    # Update the control signal with adaptation
                    u += u_adapt
                
                # Adding an external force field to the arm, if such was defined
                if self.external_force_field is not None:
                    extra_gravity = self.model.get_gravity_bias() * self.external_force_field
                    u += extra_gravity

                # Accounting for the not-moving grippers (to adapt dimensions)
                u = np.hstack((u, np.zeros(self.n_gripper_joints)))

                # Actuating the arm, calculate error and update viewer ---------------------------
                self.model.send_forces(u)           
                self.model.viewer.render()
                
                # retrieve the position of the arm follow actuation
                ee_position = self.model.get_ee_position()
                
                # Calculate error as the distance between the target and the position of the EE
                error = np.sqrt(np.sum((np.array(target[:3]) - np.array(ee_position))** 2))
                
                # Monitoring  --------------------------------------------------------------------

                self.monitor_dict[exp]['error']. append(np.copy(error))       # Error step
                self.monitor_dict[exp]['ee'].    append(np.copy(ee_position)) # Position of the EE
                self.monitor_dict[exp]['q'].     append(np.copy(position))    # Joints' angles
                self.monitor_dict[exp]['dq'].    append(np.copy(velocity))    # Joints' velocities
       
                    
        # End of simulation ----------------------------------------------------------------------
        time.sleep(1.5)
        glfw.destroy_window(self.model.viewer.window)
    
      
    # Monitoring methods -------------------------------------------------------------------------
    
    def show_monitor(self):
        """ Display monitored motion and performance of the arm"""
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        # For each specified target 
        for exp in self.monitor_dict:

            # Plot EE convergence to target -------------------------------------------------------
            print('Covering a distance of {}, with an error of: {}, in {} steps '.format(
                np.sqrt(np.sum((self.monitor_dict[exp]['target_real'] - 
                                self.monitor_dict[exp]['ee'][0])**2)), 
                self.monitor_dict[exp]['error'][-1], 
                self.monitor_dict[exp]['steps']))
            plt.figure()
            plt.ylabel("Distance (m)")
            plt.xlabel("Time (ms)")
            plt.title("Distance to target")
            plt.plot(self.monitor_dict[exp]['error'])
            plt.show()
            
            # Plot EE trajectory ------------------------------------------------------------------

            ax = plt.figure().add_subplot(111, projection='3d')
            ee_x = [ee[0] for ee in self.monitor_dict[exp]['ee']]
            ee_y = [ee[1] for ee in self.monitor_dict[exp]['ee']]
            ee_z = [ee[2] for ee in self.monitor_dict[exp]['ee']]

            ax.set_title("End-Effector Trajectory")
            ax.plot(ee_x, ee_y, ee_z)

            ax.scatter(self.monitor_dict[exp]['target_real'][0], self.monitor_dict[exp]['target_real'][1], 
                       self.monitor_dict[exp]['target_real'][2], label="target", c="r")
            ax.legend()   
