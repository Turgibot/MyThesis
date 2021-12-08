"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class contains the static robot 3d object. It collects and interfaces static data to be used in the simulation.
Note that dynamic data is collected via the dynamics.py file

"""
import mujoco_py as mjc
import os
import numpy as np

class Scene:
    def __init__(self, path_to_xml, home={0: -np.pi/2, 1:0, 2:np.pi/2, 3:0, 4:np.pi/2, 5:0}):
        self.xml = path_to_xml
        try:
            self.model = mjc.load_model_from_path(self.xml)
        except:
            print("cwd: {}".format(os.getcwd()))
            raise Exception("Mujoco failed to load MJCF file from path {}".format(self.xml))

        self.joint_ids, self.joint_names = self.get_joints_meta()
        self.joint_dict = self.get_joints_dict()
        self.n_jnts = len(self.joint_ids)
        self.home = home

    # This method is for testing purpose only
    # Show the simulation current status. No step incermenting! 
    def advance(self):
        simulation = mjc.MjSim(self.model)
        viewer = mjc.MjViewer(simulation)
        while True:
            viewer.render()

    def get_joints_meta(self):
        model = self.model
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

    def get_joints_dict(self):
        joint_positions_addr   = [self.model.get_joint_qpos_addr(name) for name in self.joint_names]
        joint_velocity_addr    = [self.model.get_joint_qvel_addr(name) for name in self.joint_names]
        joint_dict        = {} 
        for i, ii in enumerate(self.joint_ids):
            joint_dict[ii] = {'name': self.joint_names[i], 
                                   'position_address': joint_positions_addr[i], 
                                   'velocity_address': joint_velocity_addr[i]}

        if not np.all(np.array(self.model.jnt_type)==3): # 3 stands for revolute joint
            raise Exception('Revolute joints are assumed')
        
        return joint_dict
    

    