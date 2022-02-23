"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

Operational space controller is based on:
Khatib, Oussama. 
"A unified approach for motion and force control of robot manipulators: The operational space formulation." 
IEEE Journal on Robotics and Automation 3.1 (1987): 43-53.

"""

import numpy as np

class OSC:
    
    def __init__(self, model,
                 k_v   = 20,           # Gain factor for velocity
                 k_p   = 200,          # Gain factor for position
                 k_o   = 200,          # Gain factor for orientation
                 k_i   = 0,            # Gain factor for integrated error
                 v_max = [0.5, 0]      # Velocity limitation for position, and orientation 
                ):
        
        self.model        = model
        self.control_dict =  {'Kv': k_v, 'Kp': k_p, 'Ko': k_o, 'Ki': k_i, 'vmax': v_max, 
                              'n_joints': self.model.n_joints}   
        
        # Same Kp for all three dimension, and same Ko for the three orientation angles
        self.control_dict['task_space_gains'] = np.array([self.control_dict['Kp']] * 3 + [self.control_dict['Ko']] * 3)           
        
        # Scaling factors for velocity limitation (see self._velocity_limiting)
        self.control_dict['lamb'] = self.control_dict['task_space_gains'] / self.control_dict['Kv']
        self.control_dict['scale_xyz'] = self.control_dict['sat_gain_xyz'] = \
                        self.control_dict['vmax'][0] / self.control_dict['Kp'] * self.control_dict['Kv']
        self.control_dict['scale_abg'] = self.control_dict['sat_gain_abg'] = \
                        self.control_dict['vmax'][1] / self.control_dict['Ko'] * self.control_dict['Kv']
    
    def generate (self, q, dq, target):
        '''
        Calculated the force control signal
        
        u = -J[Mx[Kp(x-x_target]] - Kv*M*dx/dt - u_bias
        Where: 
            Fx = Mx*d2x/dt                                    : Mx transforms acceleration to force in task space 
            d2x/dt = Kp(x-x_target) + kv(dx/dt - dx/dt_target): PD controller in operation space for accelearation
            d2x/dt = Kp(x-x_target) + kv(dx/dt)               : zero velocity at target
            Here, dx/dt is in joint space no need for Mx to convert to task space. Using M to convert to force
            J*Fx                                              : Transform force in task to joint space
            u_bias is a term for external pertubation 
        '''
                                   
        # Isolate rows of Jacobian corresponding to controlled task space DOF
        # Particularly, shosing x,y,x among [x, y, z, alpha, beta, gamma] 
        J = self.model.get_Jacobian()
        control_dof = [True, True, True, False, False, False]
        J = J[control_dof]

        # Getting the inertia matrix                           
        M = self.model.get_inertia_matrix()  # inertia matrix in joint space
        Mx, M_inv = self._Mx(M=M, J=J)       # inertia matrix in task space
        
        # calculate the desired task space forces 
        u_task = np.zeros(6)

        # position controlled (orientation control TBA)
        xyz = self.model.get_ee_position()
        u_task[:3] = xyz - target[:3]

        # task space integrated error 
        integrated_error = np.zeros(6)
        if self.control_dict['Ki'] != 0:
            integrated_error += u_task
            u_task += self.control_dict['Ki'] * integrated_error

        # if max task space velocities specified, apply velocity limiting
        if self.control_dict['vmax'] is not None:
            u_task = self._velocity_limiting(u_task)
        else:
            # otherwise apply specified gains
            u_task *= self.control_dict['task_space_gains']
        
        # Isolate task space forces corresponding to controlled DOF
        u_task = u_task[control_dof]
            
        # As there's no target velocity in task space,
        # compensate for velocity in joint space (more accurate)
        u = np.zeros(self.control_dict['n_joints'])
        dq_vector = [float(dq[i]) for i in range(self.control_dict['n_joints'])]
        u = -1 * self.control_dict['Kv'] * np.dot(M, dq_vector)

        # Transform task space control signal into joint space ----------------
        u -= np.dot(J.T, np.dot(Mx, u_task))

        # Store the current control signal u for training in case
        # dynamics adaptation signal is being used
        # NOTE: do not include gravity or null controller in training signal
        self.training_signal = np.copy(u)
        
        # add in gravity term in joint space ----------------------------------
        u -= self.model.get_gravity_bias()

        return u
      
    def _Mx(self, M, J, threshold=1e-3):
        """ Generate the task-space inertia matrix """

        # calculate the inertia matrix in task space
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        if abs(np.linalg.det(Mx_inv)) >= threshold:
            # do the linalg inverse if matrix is non-singular
            # because it's faster and more accurate
            Mx = np.linalg.inv(Mx_inv)
        else:
            # using the rcond to set singular values < thresh to 0
            # singular values < (rcond * max(singular_values)) set to 0
            Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)

        return Mx, M_inv
    
    def _velocity_limiting(self, u_task): 
        """ Scale the control signal to limit the velocity of the arm 
        
        Unlimited velocities, might cause the arm to move in a non-straight line, since not
        all actuators are able to supply the required torque due to different mass distribution). 
        
        """
        
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
        
        # Since the error along each of the x,y,x dimensions is different
        # Scaling allow speed reduction along each dimension by the same ratio, to keep the EE
        # moving in a stright line
        scale = np.ones(6)
        if norm_xyz > self.control_dict['sat_gain_xyz']:
            scale[:3] *= self.control_dict['scale_xyz'] / norm_xyz
        if norm_abg > self.control_dict['sat_gain_abg']:
            scale[3:] *= self.control_dict['scale_abg'] / norm_abg

        return self.control_dict['Kv'] * scale * self.control_dict['lamb'] * u_task