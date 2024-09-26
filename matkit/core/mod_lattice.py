'''-----------------------------------------------------------------------------
                                 mod_lattice.py

 Description: Lattice specific routines

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import sys
import math
import numpy as np


# Externally installed modules
import ase
import spglib
from ase import Atom, Atoms
from ase.spacegroup import crystal

# Local imports
from matkit.core import mod_math, mod_utility

'''----------------------------------------------------------------------------
                          MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_lattice.py"

#############
### Class ###
#############


class Class_Lattice:

    class_name = "Class: Lattice"

    lattice_type = None # Types of lattices. Ex: "fcc", "bcc", "hcp" etc
    symbols = None

    # Bravais lattice parameters
    a = None 
    b = None
    c = None

    # Lattice angles
    alpha = None # Angle between b and c
    beta = None  # Angle between c and a
    gamma = None # Angle between a and b

    ##############################
    ### Parametrized constructor #
    ##############################
    def __init__(self, *args, **kwargs):

        self.lattice_type = kwargs.get('lattice_type', None)
        self.symbols = kwargs.get('symbols', None)
        self.a = kwargs.get('a', None)
        self.b = kwargs.get('b', None)
        self.c = kwargs.get('c', None)
        self.alpha = kwargs.get('alpha', None)
        self.beta = kwargs.get('beta', None)
        self.gamma = kwargs.get('gamma', None)

        if self.lattice_type == "fcc" or self.lattice_type == "bcc":
            self.b = self.a
            self.c = self.a
            self.alpha = 90.0
            self.beta = 90.0
            self.gamma = 90.0

        if self.lattice_type == "hcp":
            self.b = self.a
            if self.c is None:
                self.c = self.a * math.sqrt(8.0/3.0)
            self.alpha = 90.0
            self.beta = 90.0
            self.gamma = 120.0

        # Check if necessary arguments are passed
        mod_utility.error_check_argument_required(
            arg_val=self.lattice_type, arg_name='lattice_type',
            module=self.class_name, subroutine='class constructor')


'''-----------------------------------------------------------------------------
                                                    SUBROUTINES
-----------------------------------------------------------------------------'''
##################
### SUBROUTINE ###
##################
def get_lattice_parameters(in_atoms, symprec=1.0e-5, angle_tolerance=-1.0,
                           to_primitive=False):

    lattice, scaled_positions, numbers = spglib.standardize_cell(
        in_atoms,
        to_primitive=to_primitive,
        no_idealize=True,
        symprec=1e-5)

    # Length of lattice vectors a, b, c
    lengths = np.zeros(3)
    for i in range(0, 3):
        lengths[i] = np.linalg.norm(lattice[i])

    # Angles between the lattice vectors alpha, beta and gamma in degrees
    angles = np.zeros(3)
    for i in range(0, 3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = mod_math.abs_cap(
            np.dot(lattice[j], lattice[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi

    cellpar = np.concatenate([lengths, angles])
    return cellpar


def get_crystal_system(sgn, detailed=False):
    def f(i, j): return i <= sgn <= j
    if detailed:
        cs = {"triclinic": (1, 2), "monoclinic": (3, 15),
              "orthorhombic": (16, 74), "tetragonal-II": (75, 88),"tetragonal-I": (89, 142),
              "rhombohedral-II": (143, 148),"rhombohedral-I": (149, 167),
              "hexagonal-II": (168, 176),"hexagonal-I": (177, 194),
              "cubic-II": (195, 206), "cubic-I": (207, 230)}
    else:
        cs = {"triclinic": (1, 2), "monoclinic": (3, 15),
              "orthorhombic": (16, 74), "tetragonal": (75, 142),
              "trigonal": (143, 167), "hexagonal": (168, 194),
              "cubic": (195, 230)}

    crystal_sytem = None

    for k, v in cs.items():
        if f(*v):
            crystal_sytem = k
            break
    return crystal_sytem


    dataset = spglib.get_symmetry_dataset(in_atoms, symprec=symprec,
                                          angle_tolerance=angle_tolerance,
                                          hall_number=0)
    sgn = dataset['number']
    lattice_system = get_crystal_system(sgn, detailed=detailed)

    return (sgn, lattice_system)

##################
### SUBROUTINE ###
##################


def crystal(symbols, spacegroup, cellpar):

    if spacegroup == 225:
        return ase.spacegroup.crystal(symbols, [(0,0,0)], spacegroup=225, cellpar=cellpar)

    elif spacegroup == 191:
        return ase.spacegroup.crystal(symbols, [(0.0,0.0,0.0),(2.0/3,1.0/3,1.0/2),(1.0/3,2.0/3,1.0/2)],
                                      spacegroup=191, cellpar=cellpar)

    elif spacegroup == 194:
        return ase.spacegroup.crystal(symbols, [(1./3., 2./3., 3./4.)],
                                      spacegroup=194, cellpar=cellpar)

    else:
        sys.stderr.write("Error: In module 'mod_lattice.py'\n")
        sys.stderr.write("       In subroutine 'crystal'\n")
        sys.stderr.write("       Incorrect Space Group Number (SGN)\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
