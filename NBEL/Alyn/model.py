"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

Physical simulation is based the MuJoCo simulator (http://www.mujoco.org)
Using the simulator is subject to acquiring a license for MuJoCo (https://www.roboti.us/license.html)
"""

from os import path
import numpy as np
import glfw
import time

import mujoco_py as mjc

class MuJoCo_Model:
    
    def __init__(self, xml_specification, mesh_specification=None,
                 # Initial configuration of the arm null position
                 ref_angles = {0: -np.pi/2, 1:0, 2:np.pi/2, 3:-np.pi/2, 4:np.pi/2, 5:0}):
        
        self.ref_angles         = ref_angles
        self.xml_specification  = xml_specification
        self.mesh_specification = mesh_specification
        
        if not path.isfile(self.xml_specification):
            raise Exception('Missing XML specification at: {}'.format(self.xml_specification))
        
        if mesh_specification is not None:
            if not path.isdir(self.mesh_specification):
                raise Exception('Missing mesh specification at: {}'.format(self.mesh_specification))
        
        print('Arm model is specified at: {}'.format(self.xml_specification))
        
        try:
            self.mjc_model = mjc.load_model_from_path(self.xml_specification)
        except:
            raise Exception('Mujoco was unable to load the model')
                
        # Initializing joint dictionary
        joint_ids, joint_names = self.get_joints_info()
        joint_positions_addr   = [self.mjc_model.get_joint_qpos_addr(name) for name in joint_names]
        joint_velocity_addr    = [self.mjc_model.get_joint_qvel_addr(name) for name in joint_names]
        self.joint_dict        = {} 
        for i, ii in enumerate(joint_ids):
            self.joint_dict[ii] = {'name': joint_names[i], 
                                   'position_address': joint_positions_addr[i], 
                                   'velocity_address': joint_velocity_addr[i]}

        if not np.all(np.array(self.mjc_model.jnt_type)==3): # 3 stands for revolute joint
            raise Exception('Revolute joints are assumed')
            
        self.n_joints = len(self.joint_dict.items())
        
        # Initialize simulator
        self.simulation   = mjc.MjSim(self.mjc_model)
        self.viewer       = mjc.MjViewer(self.simulation)
        
        # Initialize model at reference position
        self.goto_null_position()
    
    def visualize(self):
        
        while True:
            
            if self.viewer.exit:
                break
            
            self.viewer.render()
            
        glfw.destroy_window(self.viewer.window)   
            
    def get_joints_info(self):
        
        model = self.mjc_model
        joint_ids = []
        joint_names = []
        body_id = model.body_name2id("EE")
        while model.body_parentid[body_id] != 0:
            jntadrs_start = model.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(model.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                tmp_names.append(model.joint_id2name(tmp_ids[-1]))
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = model.body_parentid[body_id]
        joint_names = joint_names[::-1]
        joint_ids = np.array(joint_ids[::-1])

        return joint_ids, joint_names  
    
    def get_ee_position(self):
        """ Retrieve the position of the End Effector (EE) """
        
        return np.copy(self.simulation.data.get_body_xpos('EE'))
    
    def get_Jacobian(self):
        """ Returns the Jacobian of the arm (from the perspective of the EE) """

        _J3NP = np.zeros(3 * self.n_joints)
        _J3NR = np.zeros(3 * self.n_joints)
        _J6N  = np.zeros((6, self.n_joints))

        joint_dyn_addrs = np.array((list(self.joint_dict.keys())))

        # Position and rotation Jacobians are 3 x N_JOINTS
        jac_indices = np.hstack(
            [joint_dyn_addrs + (ii * self.n_joints) for ii in range(3)])

        mjc.cymj._mj_jacBodyCom(
            self.mjc_model, self.simulation.data,
            _J3NP, _J3NR, self.mjc_model.body_name2id('EE')
        )

        # get the position / rotation Jacobian hstacked (1 x N_JOINTS*3)
        _J6N[:3] = _J3NP[jac_indices].reshape((3, self.n_joints))
        _J6N[3:] = _J3NR[jac_indices].reshape((3, self.n_joints))

        return np.copy(_J6N)

    def get_inertia_matrix(self):
        """ Returns the inertia matrix of the arm """                                           
                                   
        _MNN = np.zeros(self.n_joints ** 2)
        
        joint_dyn_addrs = np.array((list(self.joint_dict.keys())))                           
        self.M_indices = [
            ii * self.n_joints + jj
            for jj in joint_dyn_addrs
            for ii in joint_dyn_addrs
        ]
                                   
        # Inertia matrix stored in Data.qM in custom sparse format,
        # Convert qM to a dense matrix with mj_fullM
        mjc.cymj._mj_fullM(self.mjc_model, _MNN, self.simulation.data.qM)
        
        M = _MNN[self.M_indices]
        M = M.reshape((self.n_joints, self.n_joints))
        return np.copy(M)

    
    # Retrieve arm properties actuation methods --------------------------------------------------
    
    def get_angles(self):
        """ Returns joint angles [rad] """
        
        q = {}
        for joint in self.joint_dict:
            q[joint] = np.copy(self.simulation.data.qpos[
                               self.joint_dict[joint]['position_address']])
        return q
    
    def get_velocity(self):
        """ Returns joint velocity [rad/sec] """
        
        v = {}
        for joint in self.joint_dict:
            v[joint] = np.copy(self.simulation.data.qvel[
                               self.joint_dict[joint]['velocity_address']])
        return v
    
    def get_target(self):
        """ Returns the position and orientation of the target """
        
        xyz_target = self.simulation.data.get_body_xpos("target")
        quat_target  = self.simulation.data.get_body_xquat("target")
        euler_angles = euler_from_quaternion(quat_target)
        return np.hstack([np.copy(xyz_target), np.copy(euler_angles)])
    
    def get_gravity_bias(self):       
        """ Returns the effects of Coriolis, centrifugal, and gravitational forces """
        
        joint_dyn_addrs = np.array((list(self.joint_dict.keys())))
        g = -1 * self.simulation.data.qfrc_bias[joint_dyn_addrs]
        return g
    
    # Arm actuation methods ----------------------------------------------------------------------
    
    def goto_null_position(self):
        """ Return arm null position, specified by ref_angles """
        
        self.send_target_angles({0: 0, 1:0, 2:0, 3:0, 4:0, 5:0})
    
    def send_target_angles(self, q):
        """ Move the arm to the specified joint configuration """
        
        for j in q:
            self.simulation.data.qpos[self.joint_dict[j]['position_address']] = q[j] + self.ref_angles[j]
        self.simulation.forward() # Compute forward kinematics
        
    def send_forces(self, u):  
        """ Apply the specified torque to the robot joints """

        # Setting the forces to the specified joints (assuming array ordering)
        self.simulation.data.ctrl[:] = u[:]

        # move simulation ahead one time step
        self.simulation.step()

        # Update position of hand object
        self.simulation.data.set_mocap_pos("hand", self.get_ee_position())

        # Update orientation of hand object
        quaternion = np.copy(self.simulation.data.get_body_xquat("EE"))
        self.simulation.data.set_mocap_quat("hand", quaternion)
