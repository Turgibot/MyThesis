"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class is responsible for collecting read-only dynamic data from the robot simulation.
This data is then used to calculate forward and inverse kinematics.

"""
import numpy as np
from .utilities import *
import mujoco_py as mjc


class Robot:
    def __init__(self,
                model,      # The mujoco xml generated model
                simulation, # The mujoco simulation object
                home_configuration = [-np.pi/2, 0, np.pi/2, 0, np.pi/2, 0],
                nap_configuration = [-0.5*np.pi, -0.61*np.pi, 1.025*np.pi, 0.5*np.pi, 0.38*np.pi, 0]
                ):
        self.model = model
        self.simulation = simulation
        self.home = np.array(home_configuration)
        self.nap = np.array(nap_configuration)
        self.take_a_nap()
        # self.go_home()
        self.n_joints = 6
        self.thetas = self.simulation.data.qpos 
        self.thetas_dot = self.simulation.data.qvel
        self.ee_config = self.get_ee_config()
        self.torques = self.simulation.data.ctrl
        s0 = [0, 0, 1, 0, 0, 0]
        s1 = [0, -1, 0, 0.112, 0, 0]
        s2 = [0, -1, 0, 0.4137, 0, 0.0603]
        s3 = [0, 0, -1, 0, -0.0603, 0]
        s4 = [-1, 0, 0, 0, -0.71686502, 0]
        s5 = [0, -1, 0, 0.71686502, 0, 0.0603]
        self.Slist =np.c_[s0, s1, s2, s3, s4, s5] 

        self.M = np.array([  [0,  0, -1, -0.06030074],
                             [-1, 0,  0, -0.15024978],
                             [0,  1,  0,  0.7172],
                             [0,  0,  0,  1]])
        self.R = self.M[0:3, 0:3]
    def take_a_nap(self):
        self.simulation.data.qpos[:] = self.nap
        self.simulation.forward()

    def go_home(self):
        self.simulation.data.qpos[:] = self.home
        self.simulation.forward()

    def zero_config(self):
        self.simulation.data.qpos[:] = np.zeros(6)
        self.simulation.forward()

    def read(self):
        self.thetas = self.simulation.data.qpos
        self.thetas_dot = self.simulation.data.qvel
        self.ee_config = self.get_ee_config()
        self.torques = self.simulation.data.ctrl

    def get_ee_position(self):
        """ Retrieve the position of the End Effector (EE) """
        
        return np.copy(self.simulation.data.get_body_xpos('EE'))
    
    def get_angles(self):
        """ Returns joint angles [rad] """
        
        q = {}
        for joint in self.model.joint_dict:
            q[joint] = np.copy(self.simulation.data.qpos[
                               self.model.joint_dict[joint]['position_address']])
        return q
    
    def get_velocity(self):
        """ Returns joint velocity [rad/sec] """
        
        v = {}
        for joint in self.model.joint_dict:
            v[joint] = np.copy(self.simulation.data.qvel[
                               self.model.joint_dict[joint]['velocity_address']])
        return v
    
    def get_target(self):
        """ Returns the position and orientation of the target """
        
        xyz_target = self.simulation.data.get_body_xpos("target")
        quat_target  = self.simulation.data.get_body_xquat("target")
        euler_angles = euler_from_quaternion(quat_target)
        return np.hstack([np.copy(xyz_target), np.copy(euler_angles)])
    
    def get_ee_config(self):
        """ Returns the position and orientation of the target """
        
        xyz = self.simulation.data.get_body_xpos("EE")
        quat  = self.simulation.data.get_body_xquat("EE")
        euler_angles = euler_from_quaternion(quat)
        return np.hstack([np.copy(xyz), np.copy(euler_angles)])
    
    def get_jacobian(self):
        """ Returns the Jacobian of the arm (from the perspective of the EE) """

        _J3NP = np.zeros(3 * self.n_joints)
        _J3NR = np.zeros(3 * self.n_joints)
        _J6N  = np.zeros((6, self.model.n_joints))

        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))

        # Position and rotation Jacobians are 3 x N_JOINTS
        jac_indices = np.hstack(
            [joint_dyn_addrs + (ii * self.n_joints) for ii in range(3)])

        mjc.cymj._mj_jacBodyCom(
            self.model.mjc_model, self.simulation.data,
            _J3NP, _J3NR, self.model.mjc_model.body_name2id('EE')
        )

        # get the position / rotation Jacobian hstacked (1 x N_JOINTS*3)
        _J6N[:3] = _J3NP[jac_indices].reshape((3, self.model.n_joints))
        _J6N[3:] = _J3NR[jac_indices].reshape((3, self.model.n_joints))

        return np.copy(_J6N)

    def get_inertia_matrix(self):
        """ Returns the inertia matrix of the arm """                                           
                                   
        _MNN = np.zeros(self.n_joints ** 2)
        
        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))                           
        self.M_indices = [
            ii * self.n_joints + jj
            for jj in joint_dyn_addrs
            for ii in joint_dyn_addrs
        ]
                                   
        # stored in mjData.qM, stored in custom sparse format,
        # convert qM to a dense matrix with mj_fullM
        mjc.cymj._mj_fullM(self.model.mjc_model, _MNN, self.simulation.data.qM)
        
        M = _MNN[self.M_indices]
        M = M.reshape((self.model.n_joints, self.model.n_joints))
        return np.copy(M)
    
    def get_gravity_bias(self):       
        """ Returns the effects of Coriolis, centrifugal, and gravitational forces """
        
        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))
        g = -1 * self.simulation.data.qfrc_bias[joint_dyn_addrs]
        return g

    def get_inverse_jacobian(self):
        jac = self.get_jacobian()
        return np.linalg.pinv(jac)
       


    def get_links_positions(self):
        pos_dict = {}
        names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'EE']
        for name in names:
            pos = self.simulation.data.get_body_xpos(name)
            pos_dict[name]=pos
        return pos_dict