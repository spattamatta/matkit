'''----------------------------------------------------------------------------
                               mod_crystal.py

 Description: This module lists various crystal related classes and routines.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import sys
import copy
import math
import importlib
import numpy as np

# Externally installed modules
import spglib
from pymatgen.analysis.elasticity.strain import Strain, Deformation

# Local imports
from matkit.core import mod_math, mod_utility, mod_tensor, mod_ase, \
    mod_lattice, mod_orientations

'''----------------------------------------------------------------------------
                                 MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_crystal.py"

#############
### Array ###
#############
fcc_thompson_tetrahedron = [
    [[ 1, 1, 1], [ 2,-1,-1], [ 0,-1, 1]],
    [[ 1, 1, 1], [-1, 2,-1], [ 1, 0,-1]],
    [[ 1, 1, 1], [-1,-1, 2], [-1, 1, 0]],
    [[-1,-1, 1], [-2, 1,-1], [ 0, 1, 1]],
    [[-1,-1, 1], [ 1,-2,-1], [-1, 0,-1]],
    [[-1,-1, 1], [ 1, 1, 2], [ 1,-1, 0]],
    [[ 1,-1,-1], [ 2, 1, 1], [ 0, 1,-1]],
    [[ 1,-1,-1], [-1,-2, 1], [ 1, 0, 1]],
    [[ 1,-1,-1], [-1, 1,-2], [-1,-1, 0]],
    [[-1, 1,-1], [-2,-1, 1], [ 0,-1,-1]],
    [[-1, 1,-1], [ 1, 2, 1], [-1, 0, 1]],
    [[-1, 1,-1], [ 1,-1,-2], [ 1, 1, 0]]
    ]


#############
### Array ###
#############
# FCC - HCP Basal/Shoji-Nishimaya variants
# For each variant, FCC Plane, FCC direction 1, FCC direction 2
fcc_hcp_basal_variants = [
    [[ 1, 1, 1], [ 1,-1, 0], [ 1, 1,-2]],
    [[ 1, 1, 1], [-1, 1, 0], [-1,-1, 2]],
    [[ 1, 1, 1], [-1, 0, 1], [ 1,-2, 1]],
    [[ 1, 1, 1], [ 1, 0,-1], [-1, 2,-1]],
    [[ 1, 1, 1], [ 0, 1,-1], [-2, 1, 1]],
    [[ 1, 1, 1], [ 0,-1, 1], [ 2,-1,-1]],

    [[ 1,-1,-1], [-1,-1, 0], [-1, 1,-2]],
    [[ 1,-1,-1],  [ 1, 1, 0], [ 1,-1, 2]],
    [[ 1,-1,-1], [ 1, 0, 1], [-1,-2, 1]],
    [[ 1,-1,-1], [-1, 0,-1], [ 1, 2,-1]],
    [[ 1,-1,-1], [ 0, 1,-1], [ 2, 1, 1]],
    [[ 1,-1,-1], [ 0,-1, 1], [-2,-1,-1]],

    [[-1, 1,-1], [-1,-1, 0], [-1, 1, 2]],
    [[-1, 1,-1], [ 1, 1, 0], [ 1,-1,-2]],
    [[-1, 1,-1], [-1, 0, 1], [ 1, 2, 1]],
    [[-1, 1,-1], [ 1, 0,-1], [-1,-2,-1]],
    [[-1, 1,-1], [ 0, 1, 1], [ 2, 1,-1]],
    [[-1, 1,-1], [ 0,-1,-1], [-2,-1, 1]],

    [[-1,-1, 1], [ 1,-1, 0], [ 1, 1, 2]],
    [[-1,-1, 1], [-1, 1, 0], [-1,-1,-2]],
    [[-1,-1, 1], [ 1, 0, 1], [-1, 2, 1]],
    [[-1,-1, 1], [-1, 0,-1], [ 1,-2,-1]],
    [[-1,-1, 1], [ 0, 1, 1], [-2, 1,-1]],
    [[-1,-1, 1], [ 0,-1,-1], [ 2,-1, 1]]
    ]


#############
### Array ###
#############
# FCC - HCP Prismatic variants
# For each variant, FCC Plane, FCC direction 1, FCC direction 2
fcc_hcp_prismatic_variants = [
    [[ 1, 1, 0], [ 0, 0, 1], [ 1,-1, 0]],
    [[ 1, 1, 0], [ 0, 0,-1], [-1, 1, 0]],
    [[ 1,-1, 0], [ 0, 0, 1], [-1,-1, 0]],
    [[ 1,-1, 0], [ 0, 0,-1], [ 1, 1, 0]],

    [[-1, 1, 0], [ 0, 0, 1], [-1,-1, 0]],
    [[-1, 1, 0], [ 0, 0,-1], [ 1, 1, 0]],
    [[-1,-1, 0], [ 0, 0, 1], [ 1,-1, 0]],
    [[-1,-1, 0], [ 0, 0,-1], [-1, 1, 0]],

    [[ 1, 0, 1], [ 0, 1, 0], [-1, 0, 1]],
    [[ 1, 0, 1], [ 0,-1, 0], [ 1, 0,-1]],
    [[-1, 0, 1], [ 0, 1, 0], [-1, 0,-1]],
    [[-1, 0, 1], [ 0,-1, 0], [ 1, 0, 1]],

    [[ 1, 0,-1], [ 0, 1, 0], [-1, 0,-1]],
    [[ 1, 0,-1], [ 0,-1, 0], [ 1, 0, 1]],
    [[-1, 0,-1], [ 0, 1, 0], [-1, 0, 1]],
    [[-1, 0,-1], [ 0,-1, 0], [ 1, 0,-1]],

    [[ 0, 1, 1], [ 1, 0, 0], [ 0, 1,-1]],
    [[ 0, 1, 1], [-1, 0, 0], [ 0,-1, 1]],
    [[ 0, 1,-1], [ 1, 0, 0], [ 0,-1,-1]],
    [[ 0, 1,-1], [-1, 0, 0], [ 0, 1, 1]],

    [[ 0,-1, 1], [ 1, 0, 0], [ 0,-1,-1]],
    [[ 0,-1, 1], [-1, 0, 0], [ 0, 1, 1]],
    [[ 0,-1,-1], [ 1, 0, 0], [ 0, 1,-1]],
    [[ 0,-1,-1], [-1, 0, 0], [ 0,-1, 1]]
    ]

'''
#############
### Array ###
#############
# BCC - HCP Burger variants
# For each variant, BCC Plane || (0,0,0,1) HCP, BCC direction 1 || [1,1,-2,0] HCP,  BCC direction 2 || [-1,1,0,0]
bcc_hcp_burgers_variants = [
    [[ 1, 0, 1], [-1, 1, 1], [-1,-2, 1]],
    [[-1, 0, 1], [ 1, 1, 1], [-1, 2,-1]],
    [[ 0, 1, 1], [ 1, 1,-1], [-2, 1,-1]],
    [[ 0, 1,-1], [-1, 1, 1], [ 2, 1, 1]],
    [[ 1,-1, 0], [ 1, 1,-1], [ 1, 1, 2]],
    [[ 1, 1, 0], [-1, 1, 1], [ 1,-1, 2]],
    [[ 1, 0, 1], [ 1, 1,-1], [-1, 2, 1]],
    [[-1, 0, 1], [ 1,-1, 1], [ 1, 2, 1]],
    [[ 0, 1, 1], [ 1,-1, 1], [ 2, 1,-1]],
    [[ 0, 1,-1], [ 1, 1, 1], [ 2,-1,-1]],
    [[ 1,-1, 0], [ 1, 1, 1], [-1,-1, 2]],
    [[ 1, 1, 0], [ 1,-1, 1], [ 1,-1,-2]]
    ]

#############
### Array ###
#############
# BCC - HCP Burgers symmetry operations
bcc_hcp_burgers_sym_op = [
    [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]],
    [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]], 
    [[ 0, 0, 1], [ 1, 0, 0], [ 0, 1, 0]], 
    [[ 0, 0,-1], [-1, 0, 0], [ 0, 1, 0]], 
    [[ 0,-1, 0], [ 0, 0,-1], [ 1, 0, 0]], 
    [[ 0,-1, 0], [ 0, 0, 1], [-1, 0, 0]], 
    [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]], 
    [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]], 
    [[ 0, 0,-1], [ 1, 0, 0], [ 0,-1, 0]], 
    [[ 0, 0, 1], [-1, 0, 0], [ 0,-1, 0]], 
    [[ 0, 1, 0], [ 0, 0,-1], [-1, 0, 0]], 
    [[ 0, 1, 0], [ 0, 0, 1], [ 1, 0, 0]]
    ]
'''

#############
### Array ###
#############
# BCC - HCP Burger variants
# For each variant, BCC Plane || (0,0,0,1) HCP, BCC direction 1 || [1,-1,0,0] HCP,  BCC direction 2 || [1,1,-2,0]
# NOTE: Plane and Direction 2 are important
# Plane is z, Direction 1 is x and Direction 2 is y
bcc_hcp_burgers_variants = [
    [[ 1, 1, 0], [ 1,-1,-2], [-1, 1,-1]],
    [[ 1, 1, 0], [ 1,-1, 2], [ 1,-1,-1]],
    [[ 1,-1, 0], [-1,-1,-2], [ 1, 1,-1]],
    [[ 1,-1, 0], [ 1, 1,-2], [ 1, 1, 1]],
    [[ 0, 1, 1], [ 2,-1, 1], [ 1, 1,-1]],
    [[ 0, 1, 1], [ 2, 1,-1], [-1, 1,-1]],
    [[ 0, 1,-1], [-2,-1,-1], [-1, 1, 1]],
    [[ 0, 1,-1], [-2, 1, 1], [ 1, 1, 1]],
    [[ 1, 0, 1], [ 1, 2,-1], [-1, 1, 1]],
    [[ 1, 0, 1], [ 1,-2,-1], [ 1, 1,-1]],
    [[-1, 0, 1], [-1,-2,-1], [ 1,-1, 1]],
    [[-1, 0, 1], [ 1,-2, 1], [ 1, 1, 1]]
    ]

#############
### Class ###
#############


class Class_Orient_Specifiers:

    '''
    A class to hold the orientation specifiers (plane and directions)
    '''

    # Type of indexing: "miller" or "miller-bravais"
    indexing_type = None

    # Each row is either a direction or plane in the specified indexing notation
    spec_matrix_i = None

    # Each row is the cartesian counterpart of each row of spec_matrix
    # NOTE: All are directions, no planes
    basis_matrix_c = None

    # Each element corresponds to a row in spec_matrix i.e "plane" or "direction"
    spec_types = None

    ############################
    # Parametrized constructor #
    ############################
    def __init__(self,
                 lattice,
                 indexing_type,
                 spec_types=None,
                 spec_matrix_i=None,
                 basis_matrix_c=None):

        self.indexing_type = indexing_type

        if spec_matrix_i is not None:
            self.spec_matrix_i = copy.deepcopy(spec_matrix_i)

        if spec_types is not None:
            self.spec_types = copy.deepcopy(spec_types)

        if (spec_types is not None) and (spec_matrix_i is not None):
            if len(spec_types) != len(spec_matrix_i):
                sys.stderr.write("Error: In module %s'\n" %(module_name))
                sys.stderr.write("       In class constructor 'Class_Orient_Specifiers'\n")
                sys.stderr.write("       Length i.e # rows of spec_matrix should be equal to length of spec_types\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)
 
        if basis_matrix_c is not None:
            self.basis_matrix_c = copy.deepcopy(basis_matrix_c)

        else:
            # If basis_matrix_c is not specified, basis_matrix_i and spec_types must be specified
            if spec_types is None or spec_matrix_i is None:
                sys.stderr.write("Error: In module %s'\n" %(module_name))
                sys.stderr.write("       In class constructor 'Class_Orient_Specifiers'\n")
                sys.stderr.write("       If basis_matrix_c is not specified, then both the spec_matrix_i and spec_types must be specified\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)

            # Convert to cartesian
            self.basis_matrix_c = np.empty((0,3))
            for spec_i, spec_type in zip(spec_matrix_i, spec_types):
                dir_c = lattice_direction_to_cartesian_unit_vector(lattice, spec_i)
                self.basis_matrix_c = np.append(self.basis_matrix_c, np.array([dir_c], dtype='float64'), axis=0)

            # Complete the basis if not complete
            if len(self.basis_matrix_c) < 2:
                sys.stderr.write("Error: In module %s'\n" %(module_name))
                sys.stderr.write("       In class constructor 'Class_Orient_Specifiers'\n")
                sys.stderr.write("       Atleast two directions are required to construct the oriented basis\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)

            if len(self.basis_matrix_c) == 2:
                dir3 = np.cross(self.basis_matrix_c[0], self.basis_matrix_c[1])
                self.basis_matrix_c = np.append(self.basis_matrix_c, np.array([dir3], dtype='float64'), axis=0)

#############
### Class ###
#############


class Class_Orientation_Relation:

    '''
    g_basis -- Global cartesian basis, abbreviated as "gc"
    p_basis_gc -- Cartesian basis of crystal "p" expressed in gc
    q_basis_gc -- Cartesian basis of crystal "q" expressed in gc
    '''

    class_name = "Class: Class_Orientation_Relation"

    # Crystal basis w.r.t the global cartesian basis. NOTE: Initially undetermined
    p_basis_gc = None
    q_basis_gc = None

    # Type of the orientation relation
    orientation_relation_type = None

    # P and Q lattice objects
    p_lattice = None
    q_lattice = None 

    # Orientation directions or planes
    # Atleast two pairs of matching specifiers are needed
    p_orient_specs = None
    q_orient_specs = None

    #-------------------------#
    # Transformation matrices #
    #-------------------------#
    # Transformation matrices from crystal basis of p to its orientation basis and viceversa
    T_pc2poc = None
    T_poc2pc = None

    # Transformation matrices from crystal basis of q to its orientation basis and viceversa
    T_qc2qoc = None
    T_qoc2qc = None

    # Transformation matrices from crystal basis p to crystal basis q
    T_pc2qc = None
    T_qc2pc = None

    # Orientation label
    orientation_relation_label = None

    ##############################
    ### Parametrized constructor #
    ##############################
    def __init__(self,
                 orientation_relation_type,
                 p_lattice, q_lattice,
                 p_orient_specs, q_orient_specs):

        # All arguments are required arguments
        self.orientation_relation_type = orientation_relation_type

        self.p_lattice = copy.deepcopy(p_lattice)
        self.q_lattice = copy.deepcopy(q_lattice)

        self.p_orient_specs = copy.deepcopy(p_orient_specs)
        self.q_orient_specs = copy.deepcopy(q_orient_specs)

        self.compute_p_to_q_transformation_matrices()
        self.create_orientation_relation_label()

    ##############
    ### METHOD ###
    ##############
    def compute_p_to_q_transformation_matrices(self):

        '''
        Transformation matrix from "lattice p" to "lattice q" respecting
        orientation relationship.

        Equate cartesian reference frames from both crystals, so that the
        proper orientation relation is obtained. Solve for transformation
        matrix. i.e, Rotate ( q_orientation_basis ) to match
        p_orientation_basis.

        This is equivalent to rotating q_crystal_basis w.r.t
        p_crystal_basis i.e Rotate ( q_crystal_basis ) w.r.t p_crystal_basis.

        Now find the transformation tensor (Rotation matrix) that takes
        quantities in crystal p to crystal q and vice versa.
        '''

        self.T_pc2qc = np.matmul(np.linalg.inv(self.q_orient_specs.basis_matrix_c), self.p_orient_specs.basis_matrix_c)
        self.T_qc2pc = np.matmul(np.linalg.inv(self.p_orient_specs.basis_matrix_c), self.q_orient_specs.basis_matrix_c)

        self.T_pc2poc = mod_math.trans_mat_basis(dest=self.p_orient_specs.basis_matrix_c)
        self.T_poc2pc = np.transpose(self.T_pc2poc)

        self.T_qc2qoc = mod_math.trans_mat_basis(dest=self.q_orient_specs.basis_matrix_c)
        self.T_qoc2qc = np.transpose(self.T_qc2qoc)

    ##############
    ### METHOD ###
    ##############
    def create_orientation_relation_label(self):

        '''
        Orientation relation label
        Assumption p_orients_spec.spec_types is same as q_orient_specs.spec_types
        '''

        self.orientation_relation_label = self.orientation_relation_type + "\n"
        if self.p_orient_specs.spec_types is not None and self.q_orient_specs.spec_types is not None:

            for spec_type, p_spec, q_spec in zip(self.p_orient_specs.spec_types, self.p_orient_specs.spec_matrix_i, self.q_orient_specs.spec_matrix_i):

                # Plane parallelism
                if spec_type == "plane":
                    self.orientation_relation_label = self.orientation_relation_label + \
                        "{" + mod_utility.join_list(p_spec,", ") + "}$_\mathrm{" + self.p_lattice.lattice_type +"}$" + \
                        " || " + \
                        "{" + mod_utility.join_list(q_spec,", ") + "}$_\mathrm{" + self.q_lattice.lattice_type +"}$"

                # Direction parallelism
                if spec_type == "direction":
                    self.orientation_relation_label = self.orientation_relation_label + "\n" + \
                        "<" + mod_utility.join_list(p_spec,", ") + ">$_\mathrm{" + self.p_lattice.lattice_type +"}$" + \
                        " || " + \
                        "<" + mod_utility.join_list(q_spec,", ") + ">$_\mathrm{" + self.q_lattice.lattice_type +"}$"

'''----------------------------------------------------------------------------
                                 SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################


def get_sgn(in_atoms, symprec=1e-5):

    # Use spglib to determine the space group number sgn
    lattice = in_atoms.get_cell()
    positions = in_atoms.get_scaled_positions()
    numbers = in_atoms.get_atomic_numbers()
    cell = (lattice, positions, numbers)
    spacegroup = spglib.get_spacegroup(cell, symprec=symprec)
    dataset = spglib.get_symmetry_dataset(
        cell, symprec=symprec, angle_tolerance=-1.0, hall_number=0)
    sgn = dataset['number']
    return sgn

##################
### SUBROUTINE ###
##################
def create_ase_oriented_atoms(lattice_object, directions, size=(1,1,1), pbc=(1,1,1)):

    if lattice_object.lattice_type == "fcc":

        from ase.lattice.cubic import FaceCenteredCubic
        atoms = FaceCenteredCubic(directions=directions, latticeconstant=lattice_object.a, size=size, symbol=lattice_object.symbols[0], pbc=pbc)

    elif lattice_object.lattice_type == "bcc":

        from ase.lattice.cubic import BodyCenteredCubic
        atoms = BodyCenteredCubic(directions=directions, latticeconstant=lattice_object.a, size=size, symbol=lattice_object.symbols[0], pbc=pbc)


    elif lattice_object.lattice_type == "hcp":

        from ase.lattice.hexagonal import HexagonalClosedPacked
        atoms = HexagonalClosedPacked(directions=directions, latticeconstant={'a':lattice_object.a, 'c':lattice_object.c}, size=size, symbol=lattice_object.symbols[0], pbc=pbc)

    else:
        sys.stderr.write("Error: In module %s\n" %(module_name))
        sys.stderr.write("       In subroutine 'create_ase_lattice'\n")
        sys.stderr.write("       Unknown value for member lattice_type in lattice_object\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    return atoms

##################
### SUBROUTINE ###
##################
def direction_miller_bravais_to_miller(miller_bravais_dir):

    '''
    Convert a direction given in Miller-Bravais notation to Miller notation.
    '''

    # The direction in Miller-Bravais notation is [U, V, T, W]
    U = miller_bravais_dir[0]
    V = miller_bravais_dir[1]
    T = miller_bravais_dir[2]
    W = miller_bravais_dir[3]

    # The direction converted to Miller notation [u, v, w]
    u = 2*U + V # U - t or 2*U + V
    v = 2*V + U # V - T or 2*V + U
    w = W

    return tuple(np.array([u, v, w], dtype='int_'))

##################
### SUBROUTINE ###
##################
def direction_miller_to_miller_bravais(miller_dir):

    '''
    Convert a direction given in Miller notation to Miller-Bravais notation.
    '''

    # The direction in the Miller notation [U, V, W]
    U = miller_dir[0]
    V = miller_dir[1]
    W = miller_dir[2]
    
    # The direction converted to Miller-Bravais notation is [u, v, t, w]
    u = (1/3) * (2*U - V)
    v = (1/3) * (2*V - U)
    t = -(u + v)            # or -(1/3)*(U + V)
    w = W

    return tuple(np.array([u, v, t, w], dtype='int_'))

##################
### SUBROUTINE ###
##################
def plane_miller_bravais_to_miller(miller_bravais_plane):

    '''
    Convert a plane given in Miller-Bravais notation to Miller notation.
    '''

    # A plane in miller bravais notation is (h, k, i, l)
    h = miller_bravais_plane[0]
    k = miller_bravais_plane[1]
    i = miller_bravais_plane[2]
    l = miller_bravais_plane[3]

    # The plane converted to Miller notation (H, K, L)
    H = h
    K = k
    L = l

    return tuple(np.array([H, K, L], dtype='int_'))

##################
### SUBROUTINE ###
##################
def plane_miller_to_miller_bravais(miller_plane):

    '''
    Convert a plane given in Miller notation to Miller-Bravais notation.
    '''

    # A plane in miller notation is (H, K, L)
    H = miller_plane[0]
    K = miller_plane[1]
    L = miller_plane[2]

    # The plane converted to Miller-Bravais notation
    h = H
    k = K
    i = -(h+k)
    l = l

    return tuple(np.array([h, k, i, l], dtype='int_'))

##################
### SUBROUTINE ###
##################


def rhombohedral_to_cartesian(v, c_over_a):

    '''
    Transform a vector from rhombohedral to cartesian basis
    '''

    A = np.array([[1, -1/2, 0],
                  [0, np.sqrt(3)/2, 0],
                  [0, 0, c_over_a]], dtype='float64')

    v_cartesian = np.matmul(A, v)

    return v_cartesian

##################
### SUBROUTINE ###
##################


def lattice_direction_to_cartesian_unit_vector(lattice, direction, is_normalize=True):

    if lattice.lattice_type == "hcp":
        unit_vector_cartesian = construct_cartesian_unit_vector_from_hcp_miller_bravais_directions(
            direction, c_over_a=lattice.c/lattice.a, is_normalize=is_normalize)

    else:
        unit_vector_cartesian = construct_cartesian_unit_vector_cubic_miller_direction(direction, is_normalize=is_normalize)

    return unit_vector_cartesian

##################
### SUBROUTINE ###
##################

def direction_crystal_to_cartesian(direction, lattice, is_normalize=False):

    '''
    This module needs to be extended to work for any lattice type, start with
    modifying lattice class on par with box class in atomman
    '''

    direction = np.array(direction, dtype='float64')

    # Sanity check
    if math.isclose(np.linalg.norm(direction), 0.0, abs_tol=1.0e-3):
        sys.stderr.write("Error: In module %s\n" %(module_name))
        sys.stderr.write("       In subroutine 'direction_miller_bravais_to_cartesian'\n")
        sys.stderr.write("       Zero norm 'direction' is invalid\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # If in 4-index notation
    if direction.shape[-1] == 4:

        # Check if proper lattice is passed
        if lattice.lattice_type != "hcp":

            sys.stderr.write("Error: In module %s\n" %(module_name))
            sys.stderr.write("       In subroutine 'direction_crystal_to_carteisan'\n")
            sys.stderr.write("       Miller-Bravais indices given with non hexagonal lattice\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # STEP 1: Convert miller_bravais directions to miller directions
        #         NOTE: These Miller indices are still in the Hexagonal system
        dir_miller = direction_miller_bravais_to_miller(direction)

        # STEP 2: Convert rhombohedral to cartesian
        dir_cartesian = rhombohedral_to_cartesian(dir_miller, c_over_a=lattice.c / lattice.a)

    elif direction.shape[-1] == 3:

        dir_cartesian = direction

    else:
        sys.stderr.write("Error: In module %s\n" %(module_name))
        sys.stderr.write("       In subroutine 'direction_crystal_to_carteisan'\n")
        sys.stderr.write("       Miller or miller-bravais directions should be of dimension 3 or 4\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # STEP 3: Normalize if needed
    if is_normalize:
        return mod_math.l2_normalize_1d_np_vec(dir_cartesian)
    else:
        return dir_cartesian

##################
### SUBROUTINE ###
##################


def construct_cartesian_unit_vector_from_hcp_miller_bravais_directions(
        dir_miller_bravais, c_over_a, is_normalize=True):

    # STEP 1: Convert miller_bravais directions to miller directions
    #         NOTE: These Miller indices are still in the Hexagonal system
    dir_miller = direction_miller_bravais_to_miller(dir_miller_bravais)

    # STEP 2: Convert rhombohedral to cartesian
    dir_cartesian = rhombohedral_to_cartesian(dir_miller, c_over_a)

    # STEP 3: Normalize if needed
    if is_normalize:
        unit_vector_cartesian = mod_math.l2_normalize_1d_np_vec(dir_cartesian)
        return unit_vector_cartesian
    else:
        return dir_cartesian

##################
### SUBROUTINE ###
##################


def construct_cartesian_basis_from_hcp_miller_bravais_directions(
        dir_1_miller_bravais, dir_2_miller_bravais, c_over_a):

    '''
    Construct orthonormal basis from two normal directions in Miller-Bravais
    notation.
    '''
    x1 = construct_cartesian_unit_vector_from_hcp_miller_bravais_directions(
        dir_1_miller_bravais, c_over_a)
    y1 = construct_cartesian_unit_vector_from_hcp_miller_bravais_directions(
        dir_2_miller_bravais, c_over_a)
    z1 = np.cross(x1, y1)

    return np.array([x1, y1, z1], dtype='float64')

##################
### SUBROUTINE ###
##################


def construct_cartesian_unit_vector_cubic_miller_direction(dir_miller, is_normalize=True):

    if is_normalize:
        unit_vector_cartesian = mod_math.l2_normalize_1d_np_vec(dir_miller)
        return unit_vector_cartesian
    else:
        return dir_miller

##################
### SUBROUTINE ###
##################


def construct_cartesian_basis_cubic_miller_directions(dir_1_miller, dir_2_miller, dir_3_miller=None):

    # STEP 3: Construct orthonormal basis
    x1 = mod_math.l2_normalize_1d_np_vec(dir_1_miller)
    y1 = mod_math.l2_normalize_1d_np_vec(dir_2_miller)
    z1 = np.cross(x1, y1)

    return np.array([x1, y1, z1], dtype='float64')

##################
### SUBROUTINE ###
##################
def get_orientation_specs(p_lattice, q_lattice, orientation_relation_type, variant_idx=0):

    #------------------------------------------#
    # FCC-HCP Basal orientation relationship #
    #------------------------------------------#
    if orientation_relation_type == "fcc-hcp-basal":

        # ASSUMPTION: p_lattice is FCC, q_lattice is HCP
        p_indexing_type = "miller"
        q_indexing_type = "miller-bravais"

        p_spec_types = ["direction", "direction", "plane"]
        q_spec_types = ["direction", "direction", "plane"]

        # HCP never changes
        q_spec_matrix_i = np.array([ [ 1, 1,-2, 0], [ 1,-1, 0, 0], [ 0, 0, 0, 1] ], dtype='float64')

        # Default variant_idx = 0, index zero based.
        #-----------------------------------------#
        # FCC( 1, 1, 1) || HCP( 0, 0, 0, 1)    z  #
        # FCC[ 1,-1, 0] || HCP[ 1, 1,-2, 0]    x  #
        # FCC[ 1, 1,-2] || HCP[ 1,-1, 0, 0]    y  #
        #-----------------------------------------#
        if (variant_idx < 0) or (variant_idx >= 24):
            sys.stderr.write("Error: In module %s\n" %(module_name))
            sys.stderr.write("       In subroutine 'get_orientation_specs'\n")
            sys.stderr.write("       Fcc-hcp-basal variants take values [0,1,2,..,23]\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # Get basis directions
        p_plane = np.array(fcc_hcp_basal_variants[variant_idx][0], dtype='float64')
        p_dir_1 = np.array(fcc_hcp_basal_variants[variant_idx][1], dtype='float64')
        p_dir_2 = np.array(fcc_hcp_basal_variants[variant_idx][2], dtype='float64')
    
        # Assemble p_spec_matrix_i
        p_spec_matrix_i = np.array([p_dir_1, p_dir_2, p_plane], dtype='float64')

    #----------------------------------------#
    # HCP-FCC Basal orientation relationship #
    #----------------------------------------#
    elif orientation_relation_type == "hcp-fcc-basal":

        # ASSUMPTION: p_lattice is HCP, q_lattice is FCC
        p_indexing_type = "miller-bravais"
        q_indexing_type = "miller"

        p_spec_types = ["direction", "direction", "plane"]
        q_spec_types = ["direction", "direction", "plane"]

        # HCP never changes (just for simplicity)
        p_spec_matrix_i = np.array([ [ 1, 1,-2, 0], [ 1,-1, 0, 0], [ 0, 0, 0, 1] ], dtype='float64')

        # Default variant_idx = 0, index zero based.
        #-----------------------------------------#
        # FCC( 1, 1, 1) || HCP( 0, 0, 0, 1)    z  #
        # FCC[ 1,-1, 0] || HCP[ 1, 1,-2, 0]    x  #
        # FCC[ 1, 1,-2] || HCP[ 1,-1, 0, 0]    y  #
        #-----------------------------------------#
        if (variant_idx < 0) or (variant_idx >= 24):
            sys.stderr.write("Error: In module %s\n" %(module_name))
            sys.stderr.write("       In subroutine 'get_orientation_specs'\n")
            sys.stderr.write("       Hcp-fcc-basal variants take values [0,1,2,..,23]\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # Get basis directions
        q_plane = np.array(fcc_hcp_basal_variants[variant_idx][0], dtype='float64')
        q_dir_1 = np.array(fcc_hcp_basal_variants[variant_idx][1], dtype='float64')
        q_dir_2 = np.array(fcc_hcp_basal_variants[variant_idx][2], dtype='float64')
    
        # Assemble p_spec_matrix_i
        q_spec_matrix_i = np.array([q_dir_1, q_dir_2, q_plane], dtype='float64')

    #--------------------------------#
    # FCC HCP Prismatic Relationship #
    #--------------------------------#
    elif orientation_relation_type == "fcc-hcp-prismatic" or \
           orientation_relation_type == "fcc-hcp-prismatic-free":

        # ASSUMPTION: p_lattice is FCC, q_lattice is HCP
        p_indexing_type = "miller"
        q_indexing_type = "miller-bravais"

        p_spec_types = ["direction", "direction", "plane"]
        q_spec_types = ["direction", "direction", "plane"]

        # HCP never changes
        q_spec_matrix_i = np.array([ [ 0, 0, 0, 1], [ 1,-1, 1, 0], [ 1, 0, -1, 0] ], dtype='float64')

        #----------------------------------------#
        # FCC( 1, 1, 0) || HCP( 1, 0,-1, 0)   z  #
        # FCC[ 0, 0, 1] || HCP[ 0, 0, 0, 1]   x  #
        # FCC[ 1,-1, 0] || HCP[ 1,-2, 1, 0]   y  #
        #----------------------------------------#
        if (variant_idx < 0) or (variant_idx >= 24):
            sys.stderr.write("Error: In module %s\n" %(module_name))
            sys.stderr.write("       In subroutine 'get_orientation_specs'\n")
            sys.stderr.write("       Prismatic variants take values [0,1,2.., 23]\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # Get basis directions
        p_plane = np.array(fcc_hcp_prismatic_variants[variant_idx][0], dtype='float64')
        p_dir_1 = np.array(fcc_hcp_prismatic_variants[variant_idx][1], dtype='float64')
        p_dir_2 = np.array(fcc_hcp_prismatic_variants[variant_idx][2], dtype='float64')

        # Assemble p_spec_matrix_i
        p_spec_matrix_i = np.array([p_dir_1, p_dir_2, p_plane], dtype='float64')

    #------------------------------------------#
    # BCC-HCP Burgers orientation relationship #
    #------------------------------------------#
    elif orientation_relation_type == "bcc-hcp-burgers":

        # ASSUMPTION: p_lattice is BCC, q_lattice is HCP
        p_indexing_type = "miller"
        q_indexing_type = "miller-bravais"

        p_spec_types = ["direction", "direction", "plane"]
        q_spec_types = ["direction", "direction", "plane"]

        # HCP never changes
        q_spec_matrix_i = np.array([ [ 1,-1, 0, 0], [ 1, 1,-2, 0], [ 0, 0, 0, 1] ], dtype='float64')

        #-----------------------------------------#
        # ( 1, 1, 0) BCC || ( 0, 0, 0, 1) HCP  z  #
        # [ 1,-1,-2] BCC || [ 1,-1, 0, 0] HCP  x  #
        # [-1, 1,-1] BCC || [ 1, 1,-2, 0] HCP  y  #
        #-----------------------------------------#
        if (variant_idx < 0) or (variant_idx >= 12):
            sys.stderr.write("Error: In module %s\n" %(module_name))
            sys.stderr.write("       In subroutine 'get_orientation_specs'\n")
            sys.stderr.write("       bcc-hcp-basal variants take values [0,1,2,..,11]\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # Get basis directions
        p_plane = np.array(bcc_hcp_burgers_variants[variant_idx][0], dtype='float64')
        p_dir_1 = np.array(bcc_hcp_burgers_variants[variant_idx][1], dtype='float64')
        p_dir_2 = np.array(bcc_hcp_burgers_variants[variant_idx][2], dtype='float64')
    
        # Assemble p_spec_matrix_i
        p_spec_matrix_i = np.array([p_dir_1, p_dir_2, p_plane], dtype='float64')

    else:
        sys.stderr.write("Error: In module %s\n" %(module_name))
        sys.stderr.write("       In subroutine 'get_orientation_specs'\n")
        sys.stderr.write("       Unknown 'orientation_relation_type'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    #--------------------------#
    # Create orientation specs #
    #--------------------------#
    # Orientation specifications for p
    p_orient_specs = Class_Orient_Specifiers(lattice = p_lattice,
                                             indexing_type = p_indexing_type,
                                             spec_matrix_i = p_spec_matrix_i,
                                             spec_types = p_spec_types)

    # Orientation specifications for q
    q_orient_specs = Class_Orient_Specifiers(lattice = q_lattice,
                                             indexing_type = q_indexing_type,
                                             spec_matrix_i = q_spec_matrix_i,
                                             spec_types = q_spec_types)

    return (p_orient_specs, q_orient_specs)

##################
### SUBROUTINE ###
##################


def get_all_variants_orientation_relation_arr(p_lattice, q_lattice, orientation_relation_type):

    # Set number of variants for SN or basal orientation relation
    if orientation_relation_type == "fcc-hcp-basal":

        n_var = 24

    # Set number of variants for SN or basal orientation relation
    if orientation_relation_type == "hcp-fcc-basal":

        n_var = 24

    # Set number of variants for prismatic orientation relation
    if orientation_relation_type == "fcc-hcp-prismatic" or \
           orientation_relation_type == "fcc-hcp-prismatic-free":
        n_var = 24

    # Set number of variants for burgers orientation relation
    if orientation_relation_type == "bcc-hcp-burgers":
        n_var = 12

    orient_relation_arr = []
    F_p_to_q_pc_arr = []
    for var_idx in range(0,n_var):
        [obj_orientation_relation, F_p_to_q_pc] = setup_orientation_relation(p_lattice, q_lattice, orientation_relation_type, variant_idx=var_idx)
        orient_relation_arr.append(obj_orientation_relation)
        F_p_to_q_pc_arr.append(F_p_to_q_pc)

    return (orient_relation_arr, F_p_to_q_pc_arr)

##################
### SUBROUTINE ###
##################


def setup_orientation_relation(p_lattice, q_lattice, orientation_relation_type, variant_idx=0, is_include_shear=True):

    '''
    Inputs: p_lattice --> (lattice object) for lattice p
            q_lattice --> (lattice object) for lattice q
            orientation relation_type --> Type of the orientation relation

    Outputs: obj_orientation_relation --> Orientation relation object
    '''

    ############################################
    # FCC - HCP basal orientation relationship #
    # FCC( 1, 1, 1) || HCP( 0, 0, 0, 1)    z   #
    # FCC[ 1,-1, 0] || HCP[ 1, 1,-2, 0]    x   #
    # FCC[ 1, 1,-2] || HCP[ 1,-1, 0, 0]    y   #
    ############################################
    if orientation_relation_type == "fcc-hcp-basal" :

        # ASSUMPTION: p_lattice is FCC, q_lattice is HCP
        [p_orient_specs, q_orient_specs] = get_orientation_specs(p_lattice, q_lattice, orientation_relation_type, variant_idx)

        obj_orientation_relation = Class_Orientation_Relation(
            orientation_relation_type = orientation_relation_type,
            p_lattice = p_lattice, q_lattice = q_lattice,
            p_orient_specs=p_orient_specs, q_orient_specs=q_orient_specs)

        # Transformation strain from FCC to HCP
        a_fcc = p_lattice.a
        a_hcp = q_lattice.a
        c_hcp = q_lattice.c

        F_p_to_q_pc = get_fcc_to_hcp_basal_stress_free_deformation_gradient(
            a_fcc, a_hcp, c_hcp, variant_idx=variant_idx, is_include_shear=is_include_shear)

    ############################################
    # HCp - FCC basal orientation relationship #
    # HCP( 0, 0, 0, 1) || FCC( 1, 1, 1)    z   #
    # HCP[ 1, 1,-2, 0] || FCC[ 1,-1, 0]    x   #
    # HCP[ 1,-1, 0, 0] || FCC[ 1, 1,-2]    y   #
    ############################################
    if orientation_relation_type == "hcp-fcc-basal" :

        # ASSUMPTION: p_lattice is HCP, q_lattice is FCC
        [p_orient_specs, q_orient_specs] = get_orientation_specs(p_lattice, q_lattice, orientation_relation_type, variant_idx)

        obj_orientation_relation = Class_Orientation_Relation(
            orientation_relation_type = orientation_relation_type,
            p_lattice = p_lattice, q_lattice = q_lattice,
            p_orient_specs=p_orient_specs, q_orient_specs=q_orient_specs)

        # Transformation strain from HCP to FCC
        a_hcp = p_lattice.a
        c_hcp = p_lattice.c
        a_fcc = q_lattice.a

        # STEP 1: Compute fcc to hcp deformation gradient in fcc crystal basis
        F_q_to_p_qc = get_fcc_to_hcp_basal_stress_free_deformation_gradient(
            a_fcc, a_hcp, c_hcp, variant_idx=variant_idx, is_include_shear=is_include_shear)

        # STEP 2: (REVERSE) Compute hcp to fcc deformation gradient in fcc crystal basis
        F_p_to_q_qc = np.linalg.inv(F_q_to_p_qc)

        # STEP 3: Transform the deformation gradient to hcp basis
        F_p_to_q_pc = mod_math.transform_order_2_tensor_by_transmat(T=F_p_to_q_qc, Q=obj_orientation_relation.T_qc2pc)

    ##########################################
    # Prismatic orientation relationship     #
    # FCC( 1, 1, 0) || HCP( 1, 0,-1, 0)   z  #
    # FCC[ 0, 0, 1] || HCP[ 0, 0, 0, 1]   x  #
    # FCC[ 1,-1, 0] || HCP[ 1,-2, 1, 0]   y  #
    ##########################################
    if orientation_relation_type == "fcc-hcp-prismatic" or \
           orientation_relation_type == "fcc-hcp-prismatic-free":

        # ASSUMPTION: p_lattice is FCC, q_lattice is HCP
        [p_orient_specs, q_orient_specs] = get_orientation_specs(p_lattice, q_lattice, orientation_relation_type, variant_idx)

        obj_orientation_relation = Class_Orientation_Relation(
            orientation_relation_type = orientation_relation_type,
            p_lattice = p_lattice, q_lattice = q_lattice,
            p_orient_specs=p_orient_specs, q_orient_specs=q_orient_specs)

        if orientation_relation_type == "fcc-hcp-prismatic" or \
               orientation_relation_type == "fcc-hcp-prismatic-free":

            F_p_to_q_pc = Deformation([ [ 1.0, 0.0, 0.0 ], \
                                        [ 0.0, 1.0, 0.0 ], \
                                        [ 0.0, 0.0, 1.0 ] ])

    ####################################
    # Burgers orientation relationship #
    ####################################
    if orientation_relation_type == "bcc-hcp-burgers":

        # ASSUMPTION: p_lattice is BCC, q_lattice is HCP
        [p_orient_specs, q_orient_specs] = get_orientation_specs(p_lattice, q_lattice, orientation_relation_type, variant_idx)

        obj_orientation_relation = Class_Orientation_Relation(
            orientation_relation_type = orientation_relation_type,
            p_lattice = p_lattice, q_lattice = q_lattice,
            p_orient_specs=p_orient_specs, q_orient_specs=q_orient_specs)

        # Transformation deformation gradient from BCC to HCP
        a_bcc = p_lattice.a
        a_hcp = q_lattice.a
        c_hcp = q_lattice.c

        F_p_to_q_pc = get_bcc_to_hcp_burgers_stress_free_deformation_gradient(
            a_bcc, a_hcp, c_hcp, variant_idx)

    #--------------------#
    # Return the results #
    #--------------------#
    return(obj_orientation_relation, F_p_to_q_pc)

##################
### SUBROUTINE ###
##################


def get_fcc_to_hcp_basal_stress_free_deformation_gradient(
        a_fcc, a_hcp, c_hcp, variant_idx, is_include_shear):

    '''
    Returns the FCC to HCP transformation strain for a given variant
    corresponding to the Basal/Shoji-Nishimaya mechanism.
    The FCC [1,0,0], [0,1,0], [0,0,1] crystal basis is taken as reference.

       {1,1,1} FCC  || {0,0,0,1} HCP   Z-axis of local basis
       <1,-1,0> FCC || <1,1,-2,0> HCP  X-axis of local basis
       <1,1,-2> FCC || <1,-1,0,0> HCP  Y-axis of local basis

    This transformation has two components:
      1. Dilatational strain: (a) Isotropic in {1,1,1} plane and (b) Change of
         interplanar spacing normal to {1,1,1} plane.
      2. Shear strain due to shufling to change ABCABCABC in FCC to ABABAB in
         HCP on alternate planes.

    Global coordinate system:
        Referenced to FCC
        [1,0,0] --> x1; [0,1,0] --> x2; [0,0,1] --> x3

    Local coordinate system, depends on variant
       Referenced to FCC
       <1,-1,0> --> y1; <1,1,-2> --> y2; {1,1,1} --> y3

       With respect to HCP: By virtue of Orientation relation
       y1 --> [1,1,-2,0]; y2 --> [-1,1,0,0] ; y3 --> [0,0,0,1]

    Mapping lengths of transformation:

      Isotropic in-plane strain
        a_fcc / sqrt(2) --> a_hcp

      Normal to plane
        d(1,1,1) = a_fcc/sqrt(3) --> c_hcp / 2
    '''

    # Dilatational component of deformation gradient in local coordinate system
    f_inplane = math.sqrt(2) * a_hcp / a_fcc
    f_outplane = math.sqrt(3) * c_hcp / ( 2.0 * a_fcc )
    F_d = np.array([[f_inplane, 0.0, 0.0], 
                    [0.0, f_inplane, 0.0],
                    [0.0, 0.0, f_outplane]], dtype='float64')

    # Shear component is evaluated w.r.t hcp lattice.
    # NOTE: This is approximately equal to 1/(2 * sqrt(2))
    # This is along y2 direction growing along y3
    s = a_hcp / (math.sqrt(3) * c_hcp)
    F_s = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0,   s],
                    [0.0, 0.0, 1.0]], dtype='float64')

    # Total deformation gradient
    if is_include_shear:
        F = np.matmul(F_s, F_d)
    else:
        F = F_d

    # Convert to global coordinate system of FCC
    p_plane = mod_math.l2_normalize_1d_np_vec(np.array(fcc_hcp_basal_variants[variant_idx][0], dtype='float64'))
    p_dir_1 = mod_math.l2_normalize_1d_np_vec(np.array(fcc_hcp_basal_variants[variant_idx][1], dtype='float64'))
    p_dir_2 = mod_math.l2_normalize_1d_np_vec(np.array(fcc_hcp_basal_variants[variant_idx][2], dtype='float64'))

    # src and destination basis
    basis_src = np.array([p_dir_1, p_dir_2, p_plane], dtype='float64')
    basis_dest = mod_math.cartesian_ortho_norm_basis

    # Deformation gradient in destination parent crystal basis
    F_global = mod_math.transform_order_2_tensor(T=F, dest=basis_dest, src=basis_src)

    return Deformation(F_global)

##################
### SUBROUTINE ###
##################


def get_bcc_to_hcp_burgers_stress_free_deformation_gradient(
        a_bcc, a_hcp, c_hcp, variant_idx):

    '''
    Returns the BCC to HCP transformation strain for a given variant
    corresponding to the Burgers mechanism.

       ( 1, 1, 0) BCC || ( 0, 0, 0, 1) HCP  Z-axis of local basis
       [ 1,-1,-2] BCC || [ 1,-1, 0, 0] HCP  X-axis of local basis
       [-1, 1,-1] BCC || [ 1, 1,-2, 0] HCP  Y-axis of local basis

    Global coordinate system:
        Referenced to BCC
        [1,0,0] --> x1; [0,1,0] --> x2; [0,0,1] --> x3

    Local coordinate system, constant for all variants
       Referenced to BCC
       y1 --> [ 1,-1,-2] BCC || [ 1,-1, 0, 0] HCP
       y2 --> [-1, 1,-1] BCC || [ 1, 1,-2, 0] HCP
       y3 --> ( 1, 1, 0) BCC || ( 0, 0, 0, 1) HCP


    Derivation w.r.t variant 1 (variant_idx=0)
      Lattice corresponcence
        y2 - direction: [-1, 1,-1] BCC || [ 1, 1,-2, 0] HCP
        This gives: a_bcc * math.sqrt(3) / 2 --> a_hcp

        y3 - direction: [ 1, 1, 0] BCC || [ 0, 0, 0, 1] HCP
        This gives: a_bcc * math.sqrt(2) --> c_hcp

        The other lattice correponce is non orthogonal
        [ 1,-1,-1] BCC becomes [ 1,-2, 1, 0] HCP

        [1, -1, -1] BCC makes (109.47 - 90) degress with y1
        [ 1,-2, 1, 0] HCP makes (120 - 90) degrees with y1

    '''

    # Lengths of lattice vectors
    # BCC
    b1 = a_bcc * math.sqrt(3)/2
    b2 = a_bcc * math.sqrt(3)/2
    b3 = a_bcc * math.sqrt(2)

    # HCP
    h1 = a_hcp
    h2 = a_hcp
    h3 = c_hcp

    # Express correspondence lattice vectors in y1-y2-y3 basis as columns
    # First express as rows and take transpose for readability

    # t_bcc comes out to be (109.47-90) = 10.47 degrees, but is in radians
    t_bcc = abs(math.acos(np.dot(np.array([1, -1, -1]), np.array([-1, 1,-1])) / 3.0)) - math.radians(90)
    
    G_bcc = np.array([[b1 * math.cos(t_bcc), -b1 * math.sin(t_bcc), 0.0],
                      [                               0.0,                                 b2, 0.0],
                      [                               0.0,                                0.0,  b3]])
    G_bcc = np.transpose(G_bcc)

    G_hcp = np.array([[h1 * math.cos(math.radians(30)), -h1 * math.sin(math.radians(30)), 0.0],
                      [                            0.0,                               h2, 0.0],
                      [                            0.0,                              0.0,  h3]])
    G_hcp = np.transpose(G_hcp)

    F = np.matmul(G_hcp, np.linalg.inv(G_bcc))

    # Convert to global coordinate system of BCC
    p_plane = mod_math.l2_normalize_1d_np_vec(np.array(bcc_hcp_burgers_variants[variant_idx][0], dtype='float64'))
    p_dir_1 = mod_math.l2_normalize_1d_np_vec(np.array(bcc_hcp_burgers_variants[variant_idx][1], dtype='float64'))
    p_dir_2 = mod_math.l2_normalize_1d_np_vec(np.array(bcc_hcp_burgers_variants[variant_idx][2], dtype='float64'))

    # src and destination basis
    basis_src = np.array([p_dir_1, p_dir_2, p_plane], dtype='float64')
    basis_dest = mod_math.cartesian_ortho_norm_basis

    # Deformation gradient in destination parent crystal basis of BCC
    F_global = mod_math.transform_order_2_tensor(T=F, dest=basis_src, src=basis_src)

    return F_global

##################
### SUBROUTINE ###
##################


def get_bcc_to_hcp_burgers_stress_free_deformation_gradient_old(
        a_bcc, a_hcp, c_hcp, variant_idx):

    '''
    Returns the BCC to HCP transformation strain for a given variant
    corresponding to the Burgers mechanism.

       { 1, 0, 1} BCC || { 0, 0, 0, 1} HCP  Z-axis of local basis
       <-1, 1, 1> BCC || < 1, 1,-2, 0> HCP  X-axis of local basis
       <-1,-2, 1> BCC || <-1, 1, 0, 0> HCP  Y-axis of local basis

    Global coordinate system:
        Referenced to BCC
        [1,0,0] --> x1; [0,1,0] --> x2; [0,0,1] --> x3

    Local coordinate system, constant for all variants
       Referenced to BCC
       y1 --> [ 0, 1, 0] BCC || [-1, 2,-1, 0] HCP
       y2 --> [-1, 0, 1] BCC || [-1, 0, 1, 0] HCP
       y3 --> [ 1, 0, 1] BCC || [ 0, 0, 0, 1] HCP
    '''

    #-------------------------------------------------------------
    # Deformation gradient components in local basis of variant 1
    #-------------------------------------------------------------

    # Stretch
    F0 = np.array([[a_hcp/a_bcc,                            0.0,                            0.0],
                   [        0.0, math.sqrt(3/2) * a_hcp / a_bcc,                            0.0],
                   [        0.0,                            0.0, c_hcp / (math.sqrt(2) * a_bcc)]], dtype='float64')

    # Additional rotation about axis [101] BCC || [0001] HCP
    theta = math.radians(-5.26)

    Fr = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                   [math.sin(theta),  math.cos(theta), 0.0],
                   [0.0,                          0.0, 1.0]], dtype='float64')

    # Total deformation gradient
    F = np.matmul(Fr, F0)

    F_x = 0.5 * (np.transpose(F) + F ) - np.identity(3)

    # Deformation gradient of the variant
    sym_op = np.array(bcc_hcp_burgers_sym_op[variant_idx], dtype='float64')


    F_variant_local = np.matmul(np.transpose(sym_op), np.matmul(F_x, sym_op))


    # bcc_hcp_burgers_variants



    return F_variant_local



##################
### SUBROUTINE ###
##################


def read_ori_file(ori_filename):

    # Read orientation file
    bunge_euler_orientation_2d_array = mod_utility.read_array_data(
        ori_filename, n_headers=0, separator=None, data_type="float")

    return bunge_euler_orientation_2d_array


##################
### SUBROUTINE ###
##################


def generate_rotated_basis_strains(strain_voigt, bunge_euler_orientation_2d_array):

    '''
    Given a strain in voigt notation and a list of orientations in .ori
    file, transforms the strains into each of the rotated basis.
    '''

    # Convert voigt strain to full strain
    strain_full = mod_tensor.voigt_6_to_full_3x3_strain(strain_voigt)

    # Rotated basis strains
    rotated_strain_voigt_arr = np.empty((0,6))
    for euler in bunge_euler_orientation_2d_array:

        # Transofrm the strain into the rotated basis
        R = mod_orientations.bunge_to_rotation_matrix(euler) # Rotation matrix
        T = np.transpose(R)                    # Basis transformation matrix
        rotated_strain_full = np.matmul(T, np.matmul(strain_full, np.transpose(T)))

        # Push to array in voigt form
        rotated_strain_voigt = mod_tensor.full_3x3_to_voigt_6_strain(rotated_strain_full)
        rotated_strain_voigt_arr = np.append(rotated_strain_voigt_arr,
                                             np.array([rotated_strain_voigt]),
                                             axis=0)

    return rotated_strain_voigt_arr

##################
### SUBROUTINE ###
##################


def generate_rotated_basis_stresses(stress_voigt, bunge_euler_orientation_2d_array):

    '''
    Given a stress in voigt notation and a list of orientations in .ori
    file, transforms the stresses into each of the rotated basis.
    '''

    # Convert voigt strain to full strain
    stress_full = mod_tensor.voigt_6_to_full_3x3_stress(stress_voigt)

    # Rotated basis strains
    rotated_stress_voigt_arr = np.empty((0,6))
    for euler in bunge_euler_orientation_2d_array:

        # Transofrm the strain into the rotated basis
        R = mod_orientations.bunge_to_rotation_matrix(euler) # Rotation matrix
        T = np.transpose(R)                    # Basis transformation matrix
        rotated_stress_full = np.matmul(T, np.matmul(stress_full, np.transpose(T)))

        # Push to array in voigt form
        rotated_stress_voigt = mod_tensor.full_3x3_to_voigt_6_stress(rotated_stress_full)
        rotated_stress_voigt_arr = np.append(rotated_stress_voigt_arr,
                                             np.array([rotated_stress_voigt]),
                                             axis=0)

    return rotated_stress_voigt_arr

##################
### SUBROUTINE ###
##################


def get_ref_conf_oriented_ase_atoms(p_lattice, q_lattice, reference_lattice,
                                    orientation_relation_type,
                                    p_orient_basis_i, q_orient_basis_i):

    '''
    This subroutine sets up the atoms according to the crystal directions.
    Then depending on the orientation relation between the two crystals, and
    the choosen reference configuration, computes the reference states of the
    crystal systems.
    '''

    #---------------------------#
    # Setup the crystal systems #
    #---------------------------#
    # Create ase "p" atoms with the given axes orientation
    p_atoms = create_ase_oriented_atoms(lattice_object=p_lattice,
                                        directions=p_orient_basis_i,
                                        size=(1,1,1), pbc=(1,1,1))

    # Create ase "q" atoms with the given axes orientation
    q_atoms = create_ase_oriented_atoms(lattice_object=q_lattice,
                                        directions=q_orient_basis_i,
                                        size=(1,1,1), pbc=(1,1,1))

    # Create orientation relation
    [obj_OR, F_p_to_q_poc] = setup_orientation_relation(
        p_lattice=p_lattice, q_lattice=q_lattice,
        orientation_relation_type=orientation_relation_type)

    # Constrct the reference configurations based on the choosen reference lattice
    if reference_lattice == "p":

        F_q_to_p_poc = np.linalg.inv(F_p_to_q_poc)
        F_q_to_p_qoc = F_q_to_p_poc

        # Deform q atoms to reference zero strain configuration of p
        q_atoms_def = mod_ase.defgrad_ase_atoms(q_atoms, F_q_to_p_qoc)

        return (p_atoms, q_atoms_def)

    elif reference_lattice == "q":
        print("Not yet implemented\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def calculate_oriented_strain(obj_OR, strain_voigt_arr, applied_strain_basis, output_strain_basis=None):

    '''
    Calculates the deformation gradient and strain as per the orientation relation and reference.

    Inputs: p_lattice --> Lattice object of the crystal "p"
            q_lattice --> Lattice object of the crystal "q"
            orientation_relation_type --> Type of the orientation relation
    '''

    # Arguments sanity check
    mod_utility.error_check_argument_required(
        arg_val=applied_strain_basis, arg_name="applied_strain_basis", module=module_name,
        subroutine="calculate_oriented_strain",
        valid_args=["gc", "pc", "poc", "qc", "qoc"])

    #------------------------#
    # Strain transformations #
    #------------------------#
    ori_strain_voigt_arr_p = np.empty((0,6))
    ori_strain_voigt_arr_q = np.empty((0,6))
    ori_F_arr_p = []
    ori_F_arr_q = []

    for strain_voigt in strain_voigt_arr:

        if applied_strain_basis == "poc":

            # NOTE: Strain transformations are generally not needed if strain
            #       is applied in orient frame. But better to go through the
            #       general process. F_poc is same as F_qoc

            # Strain in the oriented basis for crystal "p"
            strain_voigt_poc = strain_voigt
            strain_poc = Strain.from_voigt(strain_voigt)
            F_poc = strain_poc.get_deformation_matrix()

            # Transform applied strain from applied basis to pc
            strain_voigt_pc = mod_tensor.transform_strain_voigt_by_transmat(strain_voigt_poc, obj_OR.T_poc2pc)
            F_pc = F_poc.rotate(obj_OR.T_poc2pc)
  
            # Transform applied strain from pc to qc
            strain_voigt_qc = mod_tensor.transform_strain_voigt_by_transmat(strain_voigt_pc, obj_OR.T_pc2qc)
            F_qc = F_pc.rotate(obj_OR.T_pc2qc)

            # Transform applied strain from qc to qoc
            strain_voigt_qoc = mod_tensor.transform_strain_voigt_by_transmat(strain_voigt_qc, obj_OR.T_qc2qoc)
            F_qoc = F_qc.rotate(obj_OR.T_qc2qoc)

            if output_strain_basis is None:
                output_strain_basis = "oc"

        #----------------#
        # Push to arrays #
        #----------------#
        if output_strain_basis == "c":
            ori_strain_voigt_arr_p = np.append(ori_strain_voigt_arr_p, np.array([ strain_voigt_pc ]), axis=0)
            ori_strain_voigt_arr_q = np.append(ori_strain_voigt_arr_q, np.array([ strain_voigt_qc ]), axis=0)
            ori_F_arr_p.append(F_pc)
            ori_F_arr_q.append(F_qc)

        if output_strain_basis == "oc":
            ori_strain_voigt_arr_p = np.append(ori_strain_voigt_arr_p, np.array([ strain_voigt_poc ]), axis=0)
            ori_strain_voigt_arr_q = np.append(ori_strain_voigt_arr_q, np.array([ strain_voigt_qoc ]), axis=0)
            ori_F_arr_p.append(F_poc)
            ori_F_arr_q.append(F_qoc)

    return (ori_strain_voigt_arr_p, ori_strain_voigt_arr_q) #, ori_F_arr_p, out_F_arr_q)

##################
### SUBROUTINE ###
##################


def get_reference_atoms_and_oriented_strains(
        p_mat_filename, q_mat_filename, orientation_relation_type,
        strain_voigt_arr=None, applied_strain_basis=None):

    '''
    FUTURE: This subroutine needs to be extended in future to make it more general
    such as passing reference_lattice as input arguments etc
    '''

    # Lattice p
    p_mat = importlib.import_module(p_mat_filename)
    p_lattice = p_mat.lattice

    # Lattice q
    q_mat = importlib.import_module(q_mat_filename)
    q_lattice = q_mat.lattice

    # Create orientation relation
    [obj_OR, F_p_to_q_poc] = setup_orientation_relation(
        p_lattice=p_lattice, q_lattice=q_lattice,
        orientation_relation_type=orientation_relation_type)

    # Make p as the reference lattice
    reference_lattice = "p"

    # Create atoms in the reference configration with resprect to the reference lattice
    [p_ref_atoms, q_ref_atoms] = get_ref_conf_oriented_ase_atoms(
            p_lattice=p_lattice, q_lattice=q_lattice, reference_lattice=reference_lattice,
            orientation_relation_type=orientation_relation_type,
            p_orient_basis_i = obj_OR.p_orient_specs.spec_matrix_i,
            q_orient_basis_i = obj_OR.q_orient_specs.spec_matrix_i)

    if (strain_voigt_arr is not None) and (applied-strain_basis is not None):
        # Compute oriented strain to be applied to the reference lattices
        [strain_voigt_arr_poc, strain_voigt_arr_qoc] = calculate_oriented_strain(
            obj_OR, strain_voigt_arr, applied_strain_basis=applied_strain_basis)  # FUTURE: Include output strain basis

        # Return results
        return (p_ref_atoms, q_ref_atoms, strain_voigt_arr_poc, strain_voigt_arr_qoc)

    else:
        return (p_ref_atoms, q_ref_atoms)

##################
### SUBROUTINE ###
##################


def angle_between_crystal_directions(dir_1, dir_2, dir_1_type, dir_2_type,
                                     lattice=None, is_degree=True):

    # Sanity check
    mod_utility.error_check_argument_required(
        arg_val=dir_1_type, arg_name="dir_1_type", module=module_name,
        subroutine="compare_crystal_directions", valid_args=["crystal",
        "cartesian"])

    # Sanity check
    mod_utility.error_check_argument_required(
        arg_val=dir_2_type, arg_name="dir_2_type", module=module_name,
        subroutine="compare_crystal_directions", valid_args=["crystal",
        "cartesian"])

    dir_1 = np.array(dir_1, dtype='float64')
    if dir_1_type == "crystal":
        dir_1 = direction_crystal_to_cartesian(dir_1, lattice, is_normalize=True)

    dir_2 = np.array(dir_2, dtype='float64')
    if dir_2_type == "crystal":
        dir_2 = direction_crystal_to_cartesian(dir_2, lattice, is_normalize=True)

    cos_theta = np.dot(dir_1, dir_2)
    return mod_math.acos(cos_theta, is_degree=is_degree)

##################
### SUBROUTINE ###
##################


def structure_matrix(lattice):

    '''
    Reference: Page 57, Introduction to conventioanl transmission electron microscopy
    Transforms crystal coordinates to cartesian coordinates
    cartesian = S.dot(crystal)
    '''

    a = lattice.a
    b = lattice.b
    c = lattice.c
    alpha = math.radians(lattice.alpha)
    beta = math.radians(lattice.beta)
    gamma = math.radians(lattice.gamma)

    c_alpha = math.cos(alpha)
    s_alpha = math.sin(alpha)

    c_beta = math.cos(beta)
    s_beta = math.sin(beta)

    c_gamma = math.cos(gamma)
    s_gamma = math.sin(gamma)

    V = a * b * c * math.sqrt(1 + 2*c_alpha*c_beta*c_gamma - c_alpha**2 + \
        - c_beta**2 -c_gamma**2)

    # Create the structure tensor
    S = np.zeros((3,3))

    S[0][0] = a
    S[0][1] = b * c_gamma
    S[0][2] = c * c_beta

    S[1][1] = b * s_gamma
    S[1][2] = -c * (1/s_gamma) * (c_beta*c_gamma - c_alpha)

    S[2][2] = V / (a*b*s_gamma)

    return S

'''----------------------------------------------------------------------------
                              END OF MODULE
----------------------------------------------------------------------------'''
