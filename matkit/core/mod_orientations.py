'''----------------------------------------------------------------------------
                           mod_orientations.py

 Description: Routines related to rotation matrices.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import sys
import math
import numpy as np

# Externally installed modules
# None

# Local imports
from matkit import CONFIG
from matkit.core import mod_utility

'''----------------------------------------------------------------------------
                                  MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = 'mod_orientations.py'

'''----------------------------------------------------------------------------
                                 SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################


def is_rotation_matrix(R, tol=1.0e-6):

    '''
    Verify if a given rotation matrix is proper orthogonal.
    '''

    R_trans = np.transpose(R)
    R_trans_R_prod = np.dot(R_trans, R)
    I = np.identity(3, dtype = R.dtype)
    norm = np.linalg.norm(I - R_trans_R_prod)
    return (norm < tol)

##################
### SUBROUTINE ###
##################


def bunge_to_passive_rotation_matrix(euler, is_angle_degree=True):

    '''
    Convert Euler angles with Bunge convention to passive rotation matrix.
    The ordering of the angles is bunge euler i.e. euler = [phi1, Phi, phi2]
    '''

    mat = np.zeros([3,3], dtype='float64')

    if is_angle_degree:
        c = np.cos(np.array(euler[:])*np.pi/180.0)
        s = np.sin(np.array(euler[:])*np.pi/180.0)
    else:
        c = np.cos(euler[:])
        s = np.sin(euler[:])

    # 1st row
    mat[0, 0]  =  c[0]*c[2] - s[0]*c[1]*s[2]
    mat[0, 1]  =  s[0]*c[2] + c[0]*c[1]*s[2]
    mat[0, 2]  =  s[1]*s[2]

    # 2nd row
    mat[1, 0]  = -c[0]*s[2] - s[0]*c[1]*c[2]
    mat[1, 1]  = -s[0]*s[2] + c[0]*c[1]*c[2]
    mat[1, 2]  =  s[1]*c[2]

    # 3rd row
    mat[2, 0]  =  s[0]*s[1]
    mat[2, 1]  = -c[0]*s[1]
    mat[2, 2]  =  c[1]

    return mat

##################
### SUBROUTINE ###
##################


def bunge_to_active_rotation_matrix(euler, is_angle_degree=True):

    '''
    Active rotation matrix rotates the object in space. It is transpose of
    passive roation matrix.
    '''

    # First compute the passive rotation matrix
    R_passive = bunge_to_passive_rotation_matrix(
        euler, is_angle_degree=is_angle_degree)

    # Active rotation matrix
    R_active = np.transpose(R_passive)
    return R_active

##################
### SUBROUTINE ###
##################

def passive_rotation_matrix_to_bunge(R, is_angle_degree=True):

    '''
    Reference:  G.G. Slabaugh. Computing euler angles from a rotation matrix.
    Technical report, University of London, London, 1999
    '''

    # Sanity check: Check if rotation matrix is proper orthogonal
    if not is_rotation_matrix(R):
        sys.stderr.write("Error: In module %s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'rotation_matrix_to_bunge'\n")
        sys.stderr.write("       Rotation matrix should be proper orthogonal\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Solution: (phi1, Phi, phi2)

    Phi = math.acos(R[2,2])

    if np.isclose(Phi, 0.0, rtol=1.0e-5, atol=1.0e-8):

        phi1 = math.atan2(-R[1,0], R[0,0])
        phi2 = 0.0

    elif np.isclose(Phi, math.pi, rtol=1.0e-5, atol=1.0e-8):

        phi1 = math.atan2(R[1,0], R[0,0])
        phi2 = 0.0

    else:

        phi1 = math.atan2(R[2,0], -R[2,1])
        phi2 = math.atan2(R[0,2], R[1,2])

    if phi1 < 0.0:

        phi1 = phi1 + 2 * math.pi

    if phi2 < 0.0:

        phi2 = phi2 + 2 * math.pi

    # Convert angles to degrees
    if is_angle_degree:
        return np.array([math.degrees(phi1), math.degrees(Phi), math.degrees(phi2)])
    else:
        return np.array([phi1, Phi, phi2])

##################
### SUBROUTINE ###
##################

def active_rotation_matrix_to_bunge(R, is_angle_degree=True):

    # Passive rotation matrix
    R_passive = np.transpose(R)
    return passive_rotation_matrix_to_bunge(R=R_passive,
                                            is_angle_degree=is_angle_degree)

##################
### SUBROUTINE ###
##################


def composite_bunge_to_active_rotation_matrix(euler_2d_arr, is_angle_degree=True):

    '''
    Convert a set of rotations applied one after the other from bunge to
    active rotation matrix. The input variable euler_2d_arr contains euler
    angles in each row.

    The rotations will be applied in the order of rows, starting from 0.
    i.e mat = R(n) . R(n-1) ... R(1) . R(0)
    '''

    mat = np.eye(3, dtype='float64')
    for euler in euler_2d_arr:
        temp_mat = bunge_to_active_rotation_matrix(euler, is_angle_degree=is_angle_degree)
        mat = np.matmul(temp_mat, mat)

    return mat

##################
### SUBROUTINE ###
##################


def composite_bunge_to_bunge(euler_2d_arr, is_angle_degree=True):

    '''
    Convert a set of rotations applied one after the other to a composite
    orientation. The input variable euler_2d_arr contains euler angles in each
    row.

    The rotations will be applied in the order of rows, starting from 0.
    i.e mat = R(n) . R(n-1) ... R(1) . R(0)
    '''

    # Composite active rotation matrix
    composite_active_rotation_matrix = composite_bunge_to_active_rotation_matrix(
        euler_2d_arr, is_angle_degree=is_angle_degree)

    # Convert rotation matrix back to angles
    bunge = active_rotation_matrix_to_bunge(composite_active_rotation_matrix,
        is_angle_degree=is_angle_degree)
    
    return bunge

##################
### SUBROUTINE ###
##################


def plot_orientations(ori_filename, bunge_euler_orient_2d_array=None):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if bunge_euler_orient_2d_array is None:
        bunge_euler_orient_2d_array = mod_utility.read_array_data(ori_filename, n_headers=0, separator=None, data_type="float")

    # Plot the feasible domain
    fig = plt.figure(figsize=(8.0, 5.0))
    ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Plot an opaque sphere
    # Make data
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = 0.9 * np.outer(np.cos(u), np.sin(v))
    y = 0.9 * np.outer(np.sin(u), np.sin(v))
    z = 0.9 * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='w', rstride=1, cstride=1, alpha=1.0, linewidth=0)

    # Plot points
    ref_pt = np.array([1, 0, 0])

    for euler in bunge_euler_orient_2d_array:
        rot_mat = bunge_to_active_rotation_matrix(euler)
        pt = np.matmul(rot_mat, ref_pt)
        ax.plot([pt[0]], [pt[1]], [pt[2]], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=2, alpha=1.0)

    plt.show()

##################
### SUBROUTINE ###
##################

def plot_orientations_with_weights(ori_filename, weight_list=None, option="make_data"):

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    bunge_euler_orient_2d_array = mod_utility.read_array_data(ori_filename, n_headers=0, separator=None, data_type="float")
    n_ori = len(bunge_euler_orient_2d_array)

    # Points on sphere
    radius = 1.0
    ref_pt = np.array([radius, 0.0, 0.0])

    point_list = np.zeros((n_ori,3))
    for idx, euler in enumerate(bunge_euler_orient_2d_array, start=0):
        rot_mat = bunge_to_active_rotation_matrix(euler)
        pt = np.matmul(rot_mat, ref_pt)
        point_list[idx] = pt


    mod_utility.plot_points_weigths_on_sphere(radius=radius, point_list=point_list, weight_list=weight_list, option=option)

##################
### SUBROUTINE ###
##################


def test_01( ):

    '''
    Test rotation matrix to bunge-Euler conversion
    '''
    bunge = np.array([37.197179036517, 115.553777058624,  -76.639503389212])

    print("Original bunge angles")
    print(bunge)

    R = bunge_to_active_rotation_matrix(euler=bunge, is_angle_degree=True)
    print("Original rotation matrix")
    print(R)

    bunge_result = active_rotation_matrix_to_bunge(R, is_angle_degree=True)
    print("Bunge angles from original rotation matrix")
    print(bunge_result)

    R = bunge_to_active_rotation_matrix(euler=bunge_result, is_angle_degree=True)
    print("Rotation matrix from resultant bunge angles")
    print(R)

##################
### SUBROUTINE ###
##################


def test_02( ):

    '''
    Test composite bunge angles
    '''

    bunge_1 = np.array([37.197179036517, 115.553777058624,  -76.639503389212])
    bunge_2 = np.array([105.521003175885, 43.461191108874, -251.980488627940])

    R_composite = composite_bunge_to_active_rotation_matrix(euler_2d_arr=np.array([bunge_1, bunge_2]), is_angle_degree=True)
    print("Original composite rotation matrix")
    print(R_composite)

    bunge_compoiste = composite_bunge_to_bunge(euler_2d_arr=np.array([bunge_1, bunge_2]), is_angle_degree=True)
    R_composite = bunge_to_active_rotation_matrix(euler=bunge_compoiste, is_angle_degree=True)
    print("Resultant composite rotation matrix")
    print(R_composite)


'''----------------------------------------------------------------------------
                                 MAIN: TESTCASE
----------------------------------------------------------------------------'''
if __name__ == "__main__":

    ori_filename = CONFIG.WORK_DIR_BASE + '/Results/mtex/uniform_1000.ori'
    #plot_orientations(ori_filename)
    #test_02()

    plot_orientations_with_weights(ori_filename=ori_filename, weight_list=None)

'''----------------------------------------------------------------------------
                                 END OF MODULE
----------------------------------------------------------------------------'''
