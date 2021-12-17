
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3
class IK:
    def __init__(self):
         # Robots' joints
        self.n_joints = 6
        self.theta0 = sp.Symbol('theta0') 
        self.theta1 = sp.Symbol('theta1') 
        self.theta2 = sp.Symbol('theta2') 
        self.theta3 = sp.Symbol('theta3')
        self.theta4 = sp.Symbol('theta4')
        self.theta5 = sp.Symbol('theta5')
        
        # length of the robots' links
        self.l1 = (127-9)      * 1e-3
        self.l2 = (427-127)    * 1e-3
        self.l3 = (60)         * 1e-3
        self.l4 = (253-60)     * 1e-3
        self.l5 = (359-253)    * 1e-3
        self.l6 = (567-359)    * 1e-3

        s0 = sp.Matrix([0, 0, 1, 0, 0, 0])

        w1 = rot_axis3(self.theta0)*sp.Matrix([0, -1, 0])
        q1 = sp.Matrix([0.005, 0, 0.112])
        v1 = -1*self.cross(w1, q1)
        s1 = sp.Matrix.vstack(w1,v1)

        w2 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*sp.Matrix([0, -1, 0])
        q2 = sp.Matrix([-0.0603, 0, 0.413])
        v2 = -1*self.cross(w2, q2)
        s2 = sp.Matrix.vstack(w2,v2)

        w3 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*rot_axis2(-self.theta2)*sp.Matrix([0, 0, -1])
        q3 = sp.Matrix([-0.0603, 0, 0])
        v3 = -1*self.cross(w3, q3)
        s3 = sp.Matrix.vstack(w3,v3)
    
        w4 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*rot_axis2(-self.theta2)*rot_axis1(-self.theta3)*sp.Matrix([-1, 0, 0])
        q4 = sp.Matrix([-0.0603, 0, 0])
        v4 = -1*self.cross(w4, q4)
        s4 = sp.Matrix.vstack(w4,v4)

        w5 = rot_axis3(self.theta0)*rot_axis2(-self.theta1)*rot_axis2(-self.theta2)*rot_axis1(-self.theta3)*rot_axis2(-self.theta4)*sp.Matrix([0, -1, 0])
        q5 = sp.Matrix([-0.0603, 0, 0])
        v5 = -1*self.cross(w5, q5)
        s5 = sp.Matrix.vstack(w5,v5)


        self.space_jaco = sp.Matrix.hstack(s0, s1, s2, s3, s4,s5)


        b5 = sp.Matrix([1, 0, 0, 0, 0, 0])
        wb4 = sp.Matrix([0, -1, 0]).T*rot_axis1(self.theta5)
        qb4 = sp.Matrix([-0.1, 0, 0])
        vb4 = -1*self.cross(wb4, qb4)
        b4 = sp.Matrix.vstack(wb4.T,vb4)

        self.body_jaco = sp.Matrix.hstack(b4, b5)


    def rot_x(self):
        
        theta = sp.Symbol("theta")

        rot_x = Matrix([            [1, 0, 0, 0],
                                    [0, sp.cos(theta), -sp.sin(theta), 0],
                                    [0, sp.sin(theta), sp.cos(theta), 0], 
                                    [0, 0, 0, 1]])
       
        return rot_x       
    def cross(self, w, q):
        a = w[1]*q[2] - w[2]*q[1]
        b = w[0]*q[2] - w[2]*q[0]
        c = w[0]*q[1] - w[1]*q[0]    
        r = sp.Matrix([a, -b, c])
        return r
    def rot_y(self):
        
        theta = sp.Symbol("theta")

        
        rot_y = Matrix([       [sp.cos(theta), 0, sp.sin(theta), 0],
                                    [0,1,0, 0],
                                    [-sp.sin(theta),0,  sp.cos(theta), 0], 
                                    [0, 0, 0, 1]])
        return rot_y        

    def rot_z(self):
        
        theta = sp.Symbol("theta")

        rot_z = Matrix( [      [sp.cos(theta), -sp.sin(theta), 0, 0],
                                    [sp.sin(theta),sp.cos(theta),0, 0],
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
        return rot_z        
        
    def translate(self):
        
        x = sp.Symbol("x")
        y = sp.Symbol("y")
        z = sp.Symbol("z")

        trans = Matrix([            [1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z], 
                                    [0, 0, 0, 1]])
        return trans
    # def Tbd(self, Tsd):


ik = IK()
print(ik.body_jaco)
