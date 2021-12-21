'''
Written by Mr. Guy Tordjman 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ Dec 2021

Physical simulation is based the MuJoCo open source simulator (http://www.mujoco.org)
This class contains the kinematic function needed to calculate the arms configuration. 

'''

'''
TODO 
1. get data from robot instead - DONE
2. IK
'''
import numpy as np
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy.matrices.dense import rot_axis1, rot_axis2, rot_axis3
class Kinematics:
    def __init__(self, robot):
        self.robot = robot
         # Robots' joints
        self.theta0 = sp.Symbol('theta0') 
        self.theta1 = sp.Symbol('theta1') 
        self.theta2 = sp.Symbol('theta2') 
        self.theta3 = sp.Symbol('theta3')
        self.theta4 = sp.Symbol('theta4')
        self.theta5 = sp.Symbol('theta5')
       
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

    def htm_to_rp(self, T):
        T = np.array(T)
        return T[0: 3, 0: 3], T[0: 3, 3]
    def htm_inv(self, T):
        R, p = self.htm_to_rp(T)
        Rt = np.array(R).T
        return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

    def get_body_jacobian(self):
        b5 = sp.Matrix([1, 0, 0, 0, 0, 0])
        wb4 = sp.Matrix([0, -1, 0]).T*rot_axis1(self.theta5)
        qb4 = sp.Matrix([-0.1, 0, 0])
        vb4 = -1*self.cross(wb4, qb4)
        b4 = sp.Matrix.vstack(wb4.T,vb4)

        self.body_jaco = sp.Matrix.hstack(b4, b5)
    
    def FK(self, M, s_poe, thetas):
        T = np.array(M)
        for i in range(len(thetas) - 1, -1, -1):
            mat_exp = self.v_to_matrix_expo_form(np.array(s_poe)[:, i] * thetas[i])
            T = np.dot(self.mat_exp_to_htm(mat_exp), T)
        return T

    # def IK(self, s_poe, M, Tsd, theta0, w_err, v_err):
        # i = 0
        # end = 100
        # #get ee configuration from initial guess
        # Tsb = self.FK(M, s_poe, theta0)
        # Tbs = self.htm_inv(Tsb)
        # Tbd = np.dot(Tbs, Tsd)
    def IKinSpace(self, s_poe, M, T, thetalist0, eomg, ev):
    
        thetalist = np.array(thetalist0).copy()
        i = 0
        maxiterations = 20
        Tsb = self.FK(M,s_poe, thetalist)
        Vs = np.dot(Adjoint(Tsb),se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
            or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        while err and i < maxiterations:
            thetalist = thetalist \
                        + np.dot(np.linalg.pinv(JacobianSpace(s_poe, \
                                                            thetalist)), Vs)
            i = i + 1
            Tsb = FKinSpace(M, s_poe, thetalist)
            Vs = np.dot(Adjoint(Tsb), \
                        se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
            err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
                or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
        
        return (thetalist, not err)

    
    def Adjoint(self, T):
        R, p = self.htm_to_rp(T)
        return np.r_[np.c_[R, np.zeros((3, 3))],np.c_[np.dot(self.w_to_skew(p), R), R]]

    def trajectory(self, Tstart, Tend, iterations=100):
        Tdif = Tend - Tstart
        traj_list = []
        for i in range(1, iterations+1):
            traj_list.append(Tstart+(i/iterations)*Tdif)
        return traj_list

    def iterativeIK(self,T_target, w_err, v_err):
        t0 = self.robot.get_joints_pos()
        M = self.robot.M
        s_poe = self.robot.s_poe

        Tstart = self.FK(M, s_poe, t0)
        traj_list = self.trajectory(Tstart, T_target)
        curr_T = Tstart
        for i in range(1, len(traj_list)):
            next_T = traj_list[i]
            t0 = IKinSpace(s_poe, M, next_T, t0, w_err, v_err)[0]
            for i in range(len(t0)):
                if t0[i]>2*np.pi:
                    t0[i]%=2*np.pi
                elif t0[i]< -2*np.pi:
                    t0[i]%= -2*np.pi


        return t0

def IKinSpace(Slist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the space frame for an open chain robot
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.
    Example Input:
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([ 1.57073783,  2.99966384,  3.1415342 ]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M,Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb),se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
          or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(JacobianSpace(Slist, \
                                                          thetalist)), Vs)
        i = i + 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = np.dot(Adjoint(Tsb), \
                    se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    
    return (thetalist, not err)


def JacobianSpace(Slist, thetalist):
    """Computes the space Jacobian for an open chain robot
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The space Jacobian corresponding to the inputs (6xn real
             numbers)
    Example Input:
        Slist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]
                  [  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])
    """
    Js = np.array(Slist).copy().astype(np.float64)
    T = np.eye(4)
    for i in range(1, len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Slist)[:, i - 1] \
                                * thetalist[i - 1])))
        Js[:, i] = np.dot(Adjoint(T), np.array(Slist)[:, i])
    return Js

def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix
    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs
    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]
def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]

def TransInv(T):
    """Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix
    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R
    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def MatrixLog6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix
    :param R: A matrix in SE3
    :return: The matrix logarithm of R
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]
def se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector
    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat
    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix
    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]

def FKinSpace(M, Slist, thetalist):
    """Computes forward kinematics in the space frame for an open chain robot
    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Space Frame)
    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        T = np.dot(MatrixExp6(VecTose3(np.array(Slist)[:, i] \
                                       * thetalist[i])), T)
    return T

def VecTose3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3
    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V
    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


def MatrixExp6(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates
    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat
    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero
    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise
    Example Input:
        z = -1e-7
    Output:
        True
    """
    return abs(z) < 1e-6

def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form
    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle
    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), np.linalg.norm(expc3))

def Normalize(V):
    """Normalizes a vector
    :param V: A vector
    :return: A unit vector pointing in the same direction as z
    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / np.linalg.norm(V)


def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    """
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

