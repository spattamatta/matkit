'''----------------------------------------------------------------------------
                                 mod_tensor.py

 Description: General routines related to tensors, continuum mechanics and
              elasticity.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
---------------------------------------------------------------------------'''
# Standard python imports
import sys
import math
import scipy
import numpy as np
from scipy import linalg

# Externally installed modules
from pymatgen.analysis.elasticity.strain import Strain
from pymatgen.analysis.elasticity.stress import Stress

# Local imports
from matkit.core import mod_math

'''----------------------------------------------------------------------------
                                   SUBROUTINES
----------------------------------------------------------------------------'''

# The indices of the full stiffness matrix of (orthorhombic) interest
Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

##################
### SUBROUTINE ###
##################


def full_3x3_to_Voigt_6_index(i, j):
    if i == j:
        return i
    return 6-i-j

##################
### SUBROUTINE ###
##################


def compute_SOEC_elastic_strain_energy_density(SOEC_voigt=None,
                                               strain_voigt=None):

    stress_voigt = np.matmul(SOEC_voigt, strain_voigt)
    U = 0.5 * np.dot(stress_voigt, strain_voigt)

    return U

##################
### SUBROUTINE ###
##################


def compute_TOEC_elastic_strain_energy_density(TOEC_voigt=None,
                                               strain_voigt=None):

    U = 0.0
    for i in range(0,6):
        for j in range(0,6):
            for k in range(0,6):
                U = U + TOEC_voigt[i][j][k]*strain_voigt[i]*strain_voigt[j]*strain_voigt[k]

    return (1/6.0)*U

##################
### SUBROUTINE ###
##################


def voigt_6_to_full_3x3_strain(strain_vector):
    ''' Form a 3x3 strain matrix from a 6 component vector in Voigt notation.'''
    exx = strain_vector[0]
    eyy = strain_vector[1]
    ezz = strain_vector[2]
    eyz = 0.5 * strain_vector[3]
    exz = 0.5 * strain_vector[4]
    exy = 0.5 * strain_vector[5]

    return np.array([[exx, exy, exz],
                     [exy, eyy, eyz],
                     [exz, eyz, ezz]])

##################
### SUBROUTINE ###
##################


def voigt_6_to_full_3x3_stress(stress_vector):
    '''Form a 3x3 stress matrix from a 6 component vector in Voigt notation.'''
    sxx = stress_vector[0]
    syy = stress_vector[1]
    szz = stress_vector[2]
    syz = stress_vector[3]
    sxz = stress_vector[4]
    sxy = stress_vector[5]

    return np.array([[sxx, sxy, sxz],
                     [sxy, syy, syz],
                     [sxz, syz, szz]])

##################
### SUBROUTINE ###
##################


def full_3x3_to_voigt_6_strain(strain_matrix):
    '''Form a 6 component strain vector in Voigt notation from a 3x3 matrix.'''
    return np.transpose([strain_matrix[0, 0],
                         strain_matrix[1, 1],
                         strain_matrix[2, 2],
                         strain_matrix[1, 2] + strain_matrix[2, 1],
                         strain_matrix[0, 2] + strain_matrix[2, 0],
                         strain_matrix[0, 1] + strain_matrix[1, 0]])

##################
### SUBROUTINE ###
##################


def full_3x3_to_voigt_6_stress(stress_matrix):
    '''Form a 6 component stress vector in Voigt notation from a 3x3 matrix.'''
    return np.transpose([stress_matrix[0, 0],
                         stress_matrix[1, 1],
                         stress_matrix[2, 2],
                         (stress_matrix[1, 2] + stress_matrix[1, 2]) / 2,
                         (stress_matrix[0, 2] + stress_matrix[0, 2]) / 2,
                         (stress_matrix[0, 1] + stress_matrix[0, 1]) / 2])


##################
### SUBROUTINE ###
##################

def full_3x3_to_voigt_stress_string(stress_matrix):

    voigt_stress = full_3x3_to_voigt_6_stress(stress_matrix)
    vstr = f"{voigt_stress[0]:15.6f}" + "       " + \
           f"{voigt_stress[1]:15.6f}" + "       " + \
           f"{voigt_stress[2]:15.6f}" + "       " + \
           f"{voigt_stress[3]:15.6f}" + "       " + \
           f"{voigt_stress[4]:15.6f}" + "       " + \
           f"{voigt_stress[5]:15.6f}"

    return vstr

##################
### SUBROUTINE ###
##################


def transform_strain_voigt_by_basis(strain_voigt,
                                    dest, src = mod_math.cartesian_ortho_norm_basis):
    '''
    Transform strain tensor in voigt notation to a new coordinate system given
    the two bases
    '''

    # Convert voigt to full
    strain_full = voigt_6_to_full_3x3_strain(strain_voigt)

    # Transform tensor by basis
    transformed_strain_full = mod_math.transform_order_2_tensor_by_basis(strain_full, dest, src)
    return full_3x3_to_voigt_6_strain(transformed_strain_full)

##################
### SUBROUTINE ###
##################


def transform_stress_voigt_by_basis(stress_voigt,
                                    dest, src = mod_math.cartesian_ortho_norm_basis):
    '''
    Transform stress tensor in voigt notation to a new coordinate system given
    the two bases
    '''

    # Convert voigt to full
    stress_full = voigt_6_to_full_3x3_stress(stress_voigt)

    # Transform tensor by basis
    transformed_stress_full = mod_math.transform_order_2_tensor_by_basis(stress_full, dest, src)
    return full_3x3_to_voigt_6_stress(transformed_stress_full)

##################
### SUBROUTINE ###
##################


def transform_strain_voigt_by_transmat(strain_voigt, Q):
    '''
    Transform strain tensor in voigt notation to a new coordinate system given
    the transformation matrix
    '''

    # Convert voigt to full
    strain_full = voigt_6_to_full_3x3_strain(strain_voigt)

    # Transform tensor by basis
    transformed_strain_full = mod_math.transform_order_2_tensor_by_transmat(strain_full, Q)
    return full_3x3_to_voigt_6_strain(transformed_strain_full)

##################
### SUBROUTINE ###
##################


def transform_stress_voigt_by_transmat(stress_voigt, Q):
    '''
    Transform stress tensor in voigt notation to a new coordinate system given
    the transformation matrix
    '''

    # Convert voigt to full
    stress_full = voigt_6_to_full_3x3_stress(stress_voigt)

    # Transform tensor by basis
    transformed_stress_full = mod_math.transform_order_2_tensor_by_transmat(stress_full, Q)
    return full_3x3_to_voigt_6_stress(transformed_stress_full)

##################
### SUBROUTINE ###
##################


def cubic_to_Voigt_6x6(C11, C12, C44):
    '''Make elastic constants matrix for cubic symmetry in voigt notation.'''
    return np.array([[C11, C12, C12, 0, 0, 0],
                     [C12, C11, C12, 0, 0, 0],
                     [C12, C12, C11, 0, 0, 0],
                     [0, 0, 0, C44, 0, 0],
                     [0, 0, 0, 0, C44, 0],
                     [0, 0, 0, 0, 0, C44]])

##################
### SUBROUTINE ###
##################


def full_lagrangian_strain_to_full_engineering_strain(
        eta_matrix, tol=1.0e-10, max_iter=1000):

    '''Lagrangian strain (Green strain) to Physical (Engineering) strain tensor.'''
    # Given eta, iteratively solves the equation eta = eps + 0.5*eps^2, for eps
    norm = 1.0
    n_iter = 0
    eps_matrix = eta_matrix
    while(norm > tol):
        if n_iter > max_iter:
            sys.stderr.write("Error: In module 'mod_tensor.py'\n")
            sys.stderr.write(
                "       In subroutine 'full_lagrangian_strain_to_full_engineering_strain'\n")
            sys.stderr.write(
                "       Maximum iterations {} reached before convergence \n".format(max_iter))
            sys.stderr.write("       Terminating!!!\n")
            exit(1)
        # Iteration
        eps_matrix_new = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.0
        norm = np.linalg.norm(eps_matrix_new - eps_matrix)
        eps_matrix = eps_matrix_new
        n_iter = n_iter + 1

    return eps_matrix

##################
### SUBROUTINE ###
##################


def apply_infinitesimal_strain_to_vector(eps_matrix, vec):
    '''Apply strain to vector'''
    vec_strained = np.matmul(np.identity(3) + eps_matrix, vec)
    return vec_strained

##################
### SUBROUTINE ###
##################


def apply_defgrad_to_vector(F, vec):
    '''Apply strain to vector'''
    vec_def = np.matmul(F, vec)
    return vec_def
    
##################
### SUBROUTINE ###
##################


def apply_defgrad_to_cell(F, ref_cell):

    def_cell = np.copy(ref_cell)
    for i in range(0, len(def_cell)):
        def_cell[i] = apply_defgrad_to_vector(F, def_cell[i])

    return def_cell

##################
### SUBROUTINE ###
##################

def right_cauchy_green_deformation(def_grad):

    C = np.matmul(def_grad, np.transpose(def_grad))
    return C

##################
### SUBROUTINE ###
##################

def green_lagrange_strain_tensor(def_grad):

    C = right_cauchy_green_deformation(def_grad)
    I = np.identity(3)

    E = 0.5*(C - I)
    return E
    
##################
### SUBROUTINE ###
##################

def small_strain_tensor(def_grad):


    I = np.identity(3)

    epsilon = 0.5 * ( np.transpose(def_grad) + def_grad ) - I
    return epsilon  

##################
### SUBROUTINE ###
##################

def right_stretch(def_grad):

    from scipy.linalg import polar
    
    _, U = polar(def_grad, "right")

    return U
    
##################
### SUBROUTINE ###
##################

def rotation(def_grad):

    from scipy.linalg import polar
    
    R, _ = polar(def_grad, "right")

    return R

##################
### SUBROUTINE ###
##################

def isotropic_stress(stress_matrix):

    stress_matrix_iso = 1.0/3.0*np.trace(stress_matrix)*np.eye(3)
    return stress_matrix_iso

##################
### SUBROUTINE ###
##################

def deviatoric_stress(stress_matrix):

    # First get isotropic component of stress
    stress_matrix_iso = isotropic_stress(stress_matrix)

    # Deviatoric stress =  stress - isotropic stress
    stress_matrix_dev = stress_matrix - stress_matrix_iso

    return stress_matrix_dev

##################
### SUBROUTINE ###
##################

def stress_invariant(stress_matrix, invariant):

    # First invariant is the trace
    if invariant == 1:
        I1 = np.trace(stress_matrix)
        return I1

    # Second invariant
    if invariant == 2:
        stress_matrix_dev = deviatoric_stress(stress_matrix)
        J2 = 1.0/2.0*np.trace(np.dot(stress_matrix_dev, stress_matrix_dev))
        return J2

    # Third invariant
    if invariant == 3:
        J3 = 1.0/3.0*np.trace(\
             np.dot(stress_matrix_dev,np.dot(stress_matrix_dev, stress_matrix_dev)))
        return J3

##################
### SUBROUTINE ###
##################
'''
def mean_stress(stress_matrix):

    # Get first invariant
    I1 = stress_invariant(stress_matrix, invariant=1)

    mean_stress = 1.0/3.0 * I1

    return mean_stress
'''

##################
### SUBROUTINE ###
##################

def equivalent_stress(stress_matrix):

    # Get second invariant
    J2 = stress_invariant(stress_matrix, invariant=2)

    eqv_stress = math.sqrt(3.0 * J2)

    return eqv_stress

##################
### SUBROUTINE ###
##################

def equivalent_stress_from_principal_stresses(principal_stresses_arr):

    p = principal_stresses_arr
    eqv_stress = math.sqrt( 0.5 * ( (p[0] - p[1])**2 + (p[1] - p[2])**2 + (p[2] - p[0])**2 ) )

    return eqv_stress

##################
### SUBROUTINE ###
##################

def convert_strain_to_deformation(strain_matrix, shape="upper"):
    """
    This function converts a strain to a deformation gradient that will
    produce that strain.  Supports three methods:
    Args:
        strain (3x3 array-like): strain matrix
        shape: (string): method for determining deformation, supports
            "upper" produces an upper triangular defo
            "lower" produces a lower triangular defo
            "symmetric" produces a symmetric defo
    """
    ftdotf = 2*strain_matrix + np.eye(3)
    if shape == "upper":
        F = linalg.cholesky(ftdotf)
    elif shape == "symmetric":
        F = linalg.sqrtm(ftdotf)
    else:
        raise ValueError("shape must be \"upper\" or \"symmetric\"")
    return F

####################################
# Various strain measure functions #
####################################

def mean_strain(strain, is_voigt=False):

    if is_voigt:
        s = Strain.from_voigt(strain)
    else:
        s = strain

    F = s.get_deformation_matrix()
    smean = F.det - 1.0

    return smean

def von_mises_strain(strain, is_voigt=False):

    if is_voigt:
        s = Strain.from_voigt(strain)
    else:
        s = strain

    return s.von_mises_strain


def xx_strain(strain, is_voigt=False):

    if is_voigt:
        return strain[0]
    else:
        return strain[0][0]


def yy_strain(strain, is_voigt=False):

    if is_voigt:
        return strain[1]
    else:
        return strain[1][1]


def zz_strain(strain, is_voigt=False):

    if is_voigt:
        return strain[2]
    else:
        return strain[2][2]


def yz_strain(strain, is_voigt=False):

    if is_voigt:
        return 0.5*strain[3]
    else:
        return strain[1][2]


def xz_strain(strain, is_voigt=False):

    if is_voigt:
        return 0.5*strain[4]
    else:
        return strain[0][2]


def xy_strain(strain, is_voigt=False):

    if is_voigt:
        return 0.5*strain[5]
    else:
        return strain[0][1]

####################################
# Various stress measure functions #
####################################


def mean_stress(stress, is_voigt=False):

    if is_voigt:
        s = Stress.from_voigt(stress)
    else:
        s = stress

    return s.mean_stress

def von_mises_stress(stress, is_voigt=False):

    if is_voigt:
        s = Stress.from_voigt(stress)
    else:
        s = stress

    """
    Slight modification from pymatgen, as I2 can be -ve due to numerical precision
    Ex: For stress: [23.333333333333336, 23.333333333333336, 23.333333333333336, 0, 0, 0]
    returns the von mises stress
    """
    if not s.is_symmetric():
        raise ValueError("The stress tensor is not symmetric, Von Mises "
                             "stress is based on a symmetric stress tensor.")
    return math.sqrt(3*abs(s.dev_principal_invariants[1]))

def xx_stress(stress, is_voigt=False):

    if is_voigt:
        return stress[0]
    else:
        return stress[0][0]


def yy_stress(stress, is_voigt=False):

    if is_voigt:
        return stress[1]
    else:
        return stress[1][1]


def zz_stress(stress, is_voigt=False):

    if is_voigt:
        return stress[2]
    else:
        return stress[2][2]


def yz_stress(stress, is_voigt=False):

    if is_voigt:
        return stress[3]
    else:
        return stress[1][2]


def xz_stress(stress, is_voigt=False):

    if is_voigt:
        return stress[4]
    else:
        return stress[0][2]


def xy_stress(stress, is_voigt=False):

    if is_voigt:
        return stress[5]
    else:
        return stress[0][1]

'''----------------------------------------------------------------------------
                             END OF MODULE
----------------------------------------------------------------------------'''
