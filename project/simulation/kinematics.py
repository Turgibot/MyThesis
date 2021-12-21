'''
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)
This class contains the kinematic function needed to calculate the arms configuration. 

'''
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3
class Kinematics:
    def __init__(self, robot):
         # Robots' joints
        self.n_joints = 6
        self.theta0 = sp.Symbol('theta0') 
        self.theta1 = sp.Symbol('theta1') 
        self.theta2 = sp.Symbol('theta2') 
        self.theta3 = sp.Symbol('theta3')
        self.theta4 = sp.Symbol('theta4')
        self.theta5 = sp.Symbol('theta5')
        
        # length of the robots' links in meters
        self.base_link = 0.067
        self.l1 = 0.045
        self.offset = 0.06
        self.l2 = 0.301
        self.l3 = 0.2
        self.l4 = 0.104 #in the -y direction when in the zero configuration
        self.ee_link = 0.075  #in the -y direction when in the zero configuration
        m_x = np.array([0, -1, 0])
        m_y = np.array([0, 0, 1])
        m_z = np.array([-1, 0, 0])
        m_p = np.array([-self.offset, -0.15, self.base_link+self.l1+self.l2+self.l3+self.l4])
        self.M = np.r_[np.c_[m_x,m_y, m_z, m_p], [[0, 0, 0, 1]]]
        # angular velocities in the space form, when in the zero configuration
        self.s_w0 = [0, 0, 1]
        self.s_w1 = [0, -1, 0]
        self.s_w2 = [0, -1, 0]
        self.s_w3 = [0, 0, -1]
        self.s_w4 = [-1, 0, 0]
        self.s_w5 = [0, -1, 0]

        # joint position in the space form, in the zero configuration

        self.s_v0 = [0, 0, 0]
        self.s_v1 = [self.base_link+self.l1, 0, 0]
        self.s_v2 = [self.base_link+self.l1+self.l2, 0, self.offset]
        self.s_v3 = [0, -self.offset, 0]
        self.s_v4 = [0, -self.base_link-self.l1-self.l2-self.l3-self.l4, 0]
        self.s_v5 = [self.base_link+self.l1+self.l2+self.l3+self.l4, 0, self.offset]

        #poe in the space form
        self.poe = self.get_space_poe()
        
    def get_space_poe(self):
        s0 = np.array(self.s_w0+self.s_v0)
        s1 = np.array(self.s_w1+self.s_v1)
        s2 = np.array(self.s_w2+self.s_v2)
        s3 = np.array(self.s_w3+self.s_v3)
        s4 = np.array(self.s_w4+self.s_v4)
        s5 = np.array(self.s_w5+self.s_v5)
        return np.c_[s0, s1, s2, s3, s4, s5]

    def w_to_skew(self, w):
        return np.array([[  0  ,-w[2],  w[1]],
                         [ w[2],  0  , -w[0]],
                         [-w[1], w[0],   0 ]])
    def skew_to_w(self, skew):
        return np.array([skew[2][1], skew[0][2], skew[1][0]])

    # using the rodriguez formula
    def skew_to_rotation_mat(self, skew):
        w = self.skew_to_w(skew)
        theta = np.linalg.norm(w)
        if abs(theta)< 1e-6:
            return np.eye(3)
        skew_normed = skew/theta
        return np.eye(3)+np.sin(theta)*skew_normed + (1 - np.cos(theta)) * np.dot(skew_normed, skew_normed)

    def w_to_rotation_mat(self, w):
        skew = self.w_to_skew(w)
        return self.skew_to_rotation_mat(skew)
    
    # represents a 1*6 velocity vector v in  a 4x4 matrix for e^[v]
    def v_to_matrix_expo_form(self, V):
        skew = self.w_to_skew([V[0], V[1], V[2]])
        v = [V[3], V[4], V[5]]
        skew_v = np.c_[skew, v]
        return np.r_[skew_v, np.zeros((1, 4))]
    
    # convert a matrix expo [V] ([S] or [B]) to its 6x1 vector representation
    def matrix_expo_to_v(self, v_mat):
        return np.c_[v_mat[2][1], v_mat[0][2], v_mat[1][0]], [v_mat[0][3], v_mat[1][3], v_mat[2][3]]
    
    # converts a 4x4 se3 mat_exp [V] to a SE3 homogenious transformation matrix (htm)
    def mat_exp_to_htm(self, exp_mat):
        skew = exp_mat[0: 3, 0: 3]
        w = self.skew_to_w(skew)
        v = exp_mat[0: 3, 3]
        theta = np.linalg.norm(w)
        if abs(theta)< 1e-6:
            return np.r_[np.c_[np.eye(3), v], [[0, 0, 0, 1]]]
        
        skew_normed = skew / theta
        rotation_mat = self.w_to_rotation_mat(w)
        g = np.eye(3) * theta + (1 - np.cos(theta)) * skew_normed + (theta - np.sin(theta)) * np.dot(skew_normed,skew_normed)
        cols = np.c_[rotation_mat, np.dot(g,v)/theta]
        return np.r_[cols,
                     [[0, 0, 0, 1]]]
    def v_to_htm(self, v):
        exp_mat = self.v_to_matrix_expo_form(v)
        return self.mat_exp_to_htm(exp_mat)
    
    def cross(self, w, q):
        a = w[1]*q[2] - w[2]*q[1]
        b = w[0]*q[2] - w[2]*q[0]
        c = w[0]*q[1] - w[1]*q[0]    
        r = sp.Matrix([a, -b, c])
        return r
   
        
    def translate(self):
        
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")

        trans = Matrix([            [1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z], 
                                    [0, 0, 0, 1]])
        return trans
    
    def get_space_jaco(self):
        # joint 0
        s0 = sp.Matrix(self.s_w0 + self.s_v0)
        

        # joint 1
        w1 = rot_axis3(self.theta0)*sp.Matrix(self.s_w1)
        q1 = sp.Matrix([0, 0, self.l1])
        v1 = -1*self.cross(w1, q1)
        s1 = sp.Matrix.vstack(w1,v1)

        # Joint 2 has an ofset in the -x direction 
        w2 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*sp.Matrix(self.s_w2)
        q2 = sp.Matrix([-self.offset, 0, 0.413])
        v2 = -1*self.cross(w2, q2)
        s2 = sp.Matrix.vstack(w2,v2)

        # joint 3
        w3 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*rot_axis2(-self.theta2)*sp.Matrix(self.s_w3)
        q3 = sp.Matrix([-0.0603, 0, 0])
        v3 = -1*self.cross(w3, q3)
        s3 = sp.Matrix.vstack(w3,v3)
        
        # joint 4
        w4 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*rot_axis2(-self.theta2)*rot_axis1(-self.theta3)*sp.Matrix(self.s_w4)
        q4 = sp.Matrix([-0.0603, 0, 0])
        v4 = -1*self.cross(w4, q4)
        s4 = sp.Matrix.vstack(w4,v4)

        # joint 5
        w5 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*rot_axis2(-self.theta2)*rot_axis1(-self.theta3)*rot_axis2(-self.theta4)*sp.Matrix(self.s_w5)
        q5 = sp.Matrix([-0.0603, 0, 0])
        v5 = -1*self.cross(w5, q5)
        s5 = sp.Matrix.vstack(w5,v5)


        return sp.Matrix.hstack(s0, s1, s2, s3, s4,s5)


    def get_body_jacobian(self):
        b5 = sp.Matrix([1, 0, 0, 0, 0, 0])
        wb4 = sp.Matrix([0, -1, 0]).T*rot_axis1(self.theta5)
        qb4 = sp.Matrix([-0.1, 0, 0])
        vb4 = -1*self.cross(wb4, qb4)
        b4 = sp.Matrix.vstack(wb4.T,vb4)

        self.body_jaco = sp.Matrix.hstack(b4, b5)
    
    def FK(self, thetas):
        T = np.array(self.M)
        for i in range(len(thetas) - 1, -1, -1):
            mat_exp = self.v_to_matrix_expo_form(np.array(self.poe)[:, i] * thetas[i])
            T = np.dot(self.mat_exp_to_htm(mat_exp), T)
        return T


