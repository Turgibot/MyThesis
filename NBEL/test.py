from arm import Model, Simulation
import numpy as np

BASE_DIR = '/home/nbel/Projects/Adaptive_arm_control/'
  
model_name  = 'NBEL'
model = Model('./project/models/vx300s/vx300s.xml')

init_angles = {0: -np.pi/2, 1:0, 2:np.pi/2, 3:0, 4:np.pi/2, 5:0}
target      = [np.array([ 0.20 , 0.10,-0.10])]


simulation_ext = Simulation(model, init_angles, external_force=0,
                                  target=target, adapt=False)

simulation_ext.simulate()

'''
simulation_ext_adapt = Simulation(model, init_angles, external_force=1.5,
                                  target=target, adapt=True)                                  
simulation_ext_adapt.simulate()
'''