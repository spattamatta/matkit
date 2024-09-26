'''-----------------------------------------------------------------------------
                                    my_math.py

 Description: Math routines

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import sys
import math
import numpy as np
from scipy.interpolate import interp1d

# Externally installed modules
# None

# Local imports
from matkit.core import mod_orientations

'''----------------------------------------------------------------------------
                                   MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_math.py"

cartesian_ortho_norm_basis = np.array([ [1.0, 0.0, 0.0], \
                                        [0.0, 1.0, 0.0], \
                                        [0.0, 0.0, 1.0] ])

'''----------------------------------------------------------------------------
                                    SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################

def test_right_handedness(v1, v2, v3):

    '''
      Tests if three vectors v1, v2 and v3 forms right handed system
      NOTE: Need not be orthonormal. All it tests is if (v1 x v2) has a
      positiove projection along v3.
    '''
    
    if np.dot( np.cross(v1, v2), v3 ) > 0.0:
        return True
    else:
        return False

##################
### SUBROUTINE ###
##################

def great_circle_distance_cartesian(p1, p2, radius):

    # Distance
    d = np.linalg.norm(p2 - p1)

    # Angle underlying one hald of "d"
    phi = math.asin(0.5 * d / radius)

    # Distance on great circle
    gc_dist = 2 * phi * radius

    return gc_dist

##################
### SUBROUTINE ###
##################

def acos(cos_theta, is_degree=True):

    if math.isclose(cos_theta, 1.0, rel_tol=1e-4):
        theta_radians = 0.0
    elif math.isclose(cos_theta, -1.0, rel_tol=1e-4):
        theta_radians = math.pi
    else:
        theta_radians = math.acos(cos_theta)

    if is_degree:
        return math.degrees(theta_radians)
    else:
        return theta_radians

##################
### SUBROUTINE ###
##################


def cart2pol(x, y, is_degree=True):
    '''
    Return polar coordinates (radius, theta)
    theta in [0, 360] by default
    '''

    radius = np.sqrt(x**2 + y**2)
    theta_rad = np.arctan2(y, x)

    if theta_rad < 0.0:
        theta_rad = theta_rad + 2 * np.pi

    if is_degree:

        return(radius, np.degrees(theta_rad))

    else:
        return (radius, theta_rad)

##################
### SUBROUTINE ###
##################


def pol2cart(rho, phi, is_degree=True):

    if is_degree:
        phi_rad = np.radians(phi)
    else:
        phi_rad = phi

    x = rho * np.cos(phi_rad)
    y = rho * np.sin(phi_rad)

    return(x, y)

##################
### SUBROUTINE ###
##################


def trans_mat_basis( dest, src = cartesian_ortho_norm_basis ):
    '''
    This matrix will transform any vector represented in basis
    A to a representation in basis B
    '''
    rmat = np.zeros((3, 3))

    rmat[0][0] = np.dot(dest[0], src[0])
    rmat[0][1] = np.dot(dest[0], src[1])
    rmat[0][2] = np.dot(dest[0], src[2])

    rmat[1][0] = np.dot(dest[1], src[0])
    rmat[1][1] = np.dot(dest[1], src[1])
    rmat[1][2] = np.dot(dest[1], src[2])

    rmat[2][0] = np.dot(dest[2], src[0])
    rmat[2][1] = np.dot(dest[2], src[1])
    rmat[2][2] = np.dot(dest[2], src[2])

    return rmat

##################
### SUBROUTINE ###
##################


def rotate_basis_by_euler_angle(src, euler):

    # Compute the transformation matrix
    R = mod_orientations.bunge_to_active_rotation_matrix(euler) # Rotation matrix

    dest = np.zeros((3,3))
    for row_idx, vec in enumerate(src, start=0):
        dest[row_idx][:] = R.dot(vec)

    return dest

##################
### SUBROUTINE ###
##################


def l2_normalize_1d_np_vec(v):
    '''
    L2 normalize a vector
    '''

    norm = np.linalg.norm(v, 2)
    return np.array(v / norm)

##################
### SUBROUTINE ###
##################

# This is wrong, do not use

def l2_normalize_2d_np_array(a, axis=0):
    '''
    Computes the row wise(axis=0) or columnwise(axis=1)
    normalizations of a given 2d numpy array
    '''

    asq = a**2.0
    axis_l2_norm = np.sqrt(np.sum(asq, axis=axis))

    # Columnwise normalization
    if axis == 0:
        return a / axis_l2_norm

    # Rowwise normalization
    elif axis == 1:
        return a / axis_l2_norm[:, np.newaxis]

    else:
        sys.stderr.write("Error: In module mod_math.py'\n")
        sys.stderr.write("       In subroutine 'l2_normalize_2d_np_array'\n")
        sys.stderr.write("       Unknown axis direction specified!!!\n")
        sys.stderr.write("       Axis specified: %d\n" % (axis))
        sys.stderr.write(
            "       specify 'axis = 0' for column wise normalization\n")
        sys.stderr.write(
            "       specify 'axis = 1' for row wise normalization\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)


##################
### SUBROUTINE ###
##################


def transform_order_2_tensor( T, dest, src = cartesian_ortho_norm_basis ):
    '''
    Transform 2nd order tensor (in full form 3x3) to a new coordinate system
    '''

    # Find the transformation matrix
    Q = trans_mat_basis(dest, src)
    Tdest = Q.dot(T.dot(Q.transpose()))
    return Tdest

##################
### SUBROUTINE ###
##################


def transform_order_2_tensor_by_basis(T, dest, src = cartesian_ortho_norm_basis):
    '''
    Transform 2nd order tensor (in full form 3x3) to a new coordinate system
    NOTE: Same as above, gradually remove the above soubroutine and its dependents
    '''

    # Find the transformation matrix
    Q = trans_mat_basis(dest, src)
    return transform_order_2_tensor_by_transmat( T, Q )

##################
### SUBROUTINE ###
##################


def transform_order_2_tensor_by_transmat( T, Q ):
    '''
    Transform 2nd order tensor (in full form 3x3) to a new coordinate system
    '''

    return np.matmul(Q, np.matmul(T, np.transpose(Q)))

##################
### SUBROUTINE ###
##################


def transform_3d_vector( v, dest, src = cartesian_ortho_norm_basis ):
    '''
    Transform 3d vector to a new coordinate system
    '''

    # Find the transformation matrix
    Q = trans_mat_basis(dest, src)
    vdest = Q.dot(v)
    return vdest

##################
### SUBROUTINE ###
##################


def convert_cartesian_to_fractional_coordinates( positions, basis ):
    '''
    Convert cartesian coordianets to direct i.e fractional coordinates
    '''

    basis_trans = np.transpose(basis)
    fractional_positions = np.empty((0, 3))
    for pos in positions:
        frac_pos = np.linalg.solve(basis_trans, pos)
        fractional_positions = np.append(
            fractional_positions, np.array(
                [frac_pos]), axis=0)

    return fractional_positions

##################
### SUBROUTINE ###
##################


def abs_cap(val, max_abs_val=1):
    '''
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.

    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.

    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    '''
    return max(min(val, max_abs_val), -max_abs_val)

##################
### SUBROUTINE ###
##################


def interpolate_matrix_rows(mat, ref_vec, ref_val):

    # Convert matrix to numpy 2d array
    np_mat = np.array(mat)
    n_row = np.size(np_mat, 0)
    n_col = np.size(np_mat, 1)

    # Sanity check:
    if len(ref_vec) != n_row:
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'interpolate_matrix_rows'\n")
        sys.stderr.write("       Number of rows of 'mat' should match the "\
           "length of the reference 'col_vec'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Sanity check
    if (ref_vec[0] <= ref_val and ref_val <= ref_vec[-1]) or \
       (ref_vec[0] >= ref_val and ref_val >= ref_vec[-1]):

        pass

    else:
        sys.stderr.write("Error: In module %s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'interpolate_matrix_rows'\n")
        sys.stderr.write("       'ref_val' should be with in the range of " \
           "'ref_vec'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Resultant interpolated row will be stored in this vector
    interp_row = np.zeros(n_col)

    for col_idx in range(0, n_col):
        mat_col = np_mat[:,col_idx]
        # Fit y = f(x) i.e mat_col = f(ref_vec)
        f_interp = interp1d(ref_vec, mat_col, kind="cubic")
        interp_row[col_idx] = f_interp(ref_val)

    return interp_row

##################
### SUBROUTINE ###
##################


def interpolate_matrix_columns(mat, ref_vec, ref_val):

    # Convert matrix to numpy 2d array
    np_mat = np.array(mat)
    n_row = np.size(np_mat, 0)
    n_col = np.size(np_mat, 1)

    # Sanity check:
    if len(ref_vec) != n_col:
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'interpolate_matrix_rows'\n")
        sys.stderr.write("       Number of columns of 'mat' should match the "\
           "length of the reference 'col_vec'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Sanity check
    if (ref_vec[0] <= ref_val and ref_val <= ref_vec[-1]) or \
       (ref_vec[0] >= ref_val and ref_val >= ref_vec[-1]):

        pass

    else:
        sys.stderr.write("Error: In module %s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'interpolate_matrix_rows'\n")
        sys.stderr.write("       'ref_val' should be with in the range of " \
           "'ref_vec'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Resultant interpolated column will be stored in this vector
    interp_col = np.zeros(n_row)

    for row_idx in range(0, n_row):
        mat_row = np_mat[row_idx,:]
        # Fit y = f(x) i.e mat_col = f(ref_vec)
        f_interp = interp1d(ref_vec, mat_row, kind="cubic")
        interp_col[row_idx] = f_interp(ref_val)

    return interp_col

##################
### SUBROUTINE ###
##################

# Check if three sides can form a triangle or not  
def check_triangle_validity(a, b, c):  
      
    # check condition  
    if (a + b <= c) or (a + c <= b) or (b + c <= a) : 
        return False
    else: 
        return True   


if __name__ == "__main__":

    c60 = np.cos(np.deg2rad(60))
    s60 = np.sin(np.deg2rad(60))

    v = np.array([ 1, 4, 2 ])
    print( transform_3d_vector(v, B) )
