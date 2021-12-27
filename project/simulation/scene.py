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
import glfw
import cv2
from .markers import Arrow
from .utilities import *
class Mujocoation:
    def __init__(self, path_to_xml):
        self.xml = path_to_xml
        try:
            self.model = mjc.load_model_from_path(self.xml)
        except:
            print("cwd: {}".format(os.getcwd()))
            raise Exception("Mujoco failed to load MJCF file from path {}".format(self.xml))
        self.simulation = mjc.MjSim(self.model)
        self.viewer = mjc.MjViewer(self.simulation)
        self.cam = self.viewer.cam
        self.cam.distance = 2
        self.cam.azimuth = -90
        self.cam.elevation = -2
        self.cam.lookat[:] = [0, 0, 0.2]
        self.offscreen = None
        self.flag = True


    # This method is for testing purpose only
    # Show the simulation current status. No step incermenting! 
    def advance_once(self):
        while True:
            self.add_arrows()
            self.viewer.render()
    
    def show_step(self):
        self.simulation.step()
        self.add_arrows()
        self.viewer.render()
        if self.flag:
            self.offscreen = mjc.MjRenderContextOffscreen(self.simulation, 0)
            self.offscreen.render(1920, 1080, 1)
            rgb = self.offscreen.read_pixels(1920, 1080)[1]
            rgb =  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb = cv2.flip(rgb, 0)
            cv2.imwrite("/home/guy/Pictures/test_image.png", rgb)
            self.flag = False    
            # glfw.destroy_window(w)
       

    def play(self, steps = 10e10):
        counter = 0
        while steps > counter:
            self.simulation.step()
            self.add_arrows()
            self.viewer.render()
            counter += 1
            

    def add_arrow(self, marker):
        self.viewer.add_marker( mat=marker.rot_mat,
                                pos=marker.pos,
                                type=100,
                                label="",
                                size=marker.size,
                                rgba=marker.rgba)

    def add_arrows(self):
        z_arrow = Arrow()
        z_arrow.set_z()
        y_arrow = Arrow()
        y_arrow.set_y()
        x_arrow = Arrow()
        x_arrow.set_x()
        self.add_arrow(x_arrow)
        self.add_arrow(y_arrow)
        self.add_arrow(z_arrow)

    def get_target_pos_euler(self):
        """ Returns the position and orientation of the target """
        
        xyz_target = self.simulation.data.get_body_xpos("target")
        quat_target  = self.simulation.data.get_body_xquat("target")
        euler_angles = euler_from_quaternion(quat_target)
        return np.copy(xyz_target), np.copy(euler_angles)
    
    def get_T_target(self):
        p, _ = self.get_target_pos_euler()
        