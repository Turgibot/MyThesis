"""
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 07 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)

This class contains the robot 3d object rendering methods and provides the needed interface
for the joints and EE position, speed and forces.

"""
import mujoco_py as mjc
import os
class VX300s:
    def __init__(self, path_to_xml):
        self.xml = path_to_xml
        try:
            self.model = mjc.load_model_from_path(self.xml)
        except:
            print("cwd: {}".format(os.getcwd()))
            raise Exception("Mujoco failed to load MJCF file from path {}".format(self.xml))

        self.simulation = mjc.MjSim(self.model)
        self.viewer = mjc.MjViewer(self.simulation)
    


    def show_simulation(self):
        while True:
            self.viewer.render()
