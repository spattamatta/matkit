'''----------------------------------------------------------------------------
                              mod_atomic_systems.py

 Description: This module creates various geometries for stacking fault,
   surface energy, surface decohesion and ideal strength simulations of various
   surface and slip systems systems (currently in fcc, bcc and hcp crystals).

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import math
import subprocess
import numpy as np

# Externally installed modules
from ase import geometry
from ase.spacegroup import crystal
from ase.build import bulk, sort
from ase.lattice.cubic import FaceCenteredCubic as fcc
from ase.lattice.cubic import BodyCenteredCubic as bcc
from ase.lattice.hexagonal import HexagonalClosedPacked as hcp
from ase.io.vasp import read_vasp, write_vasp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import generate_all_slabs

# Local imports
from matkit.core import mod_pymatgen, mod_crystal, mod_ase, mod_utility

'''----------------------------------------------------------------------------
                                 MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_atomic_systems.py"
   
'''----------------------------------------------------------------------------
                                 SUBROUTINES
----------------------------------------------------------------------------'''

dict_atomic_system = {

    'fcc_primitive' : { },
    'fcc001x100'    : {'plane' : (0,0,1), 'x' : (1,0,0),   'xy' : (0,1,0) },
    'fcc110x-110'   : {'plane' : (1,1,0), 'x' : (-1,1,0),  'xy' : (0,0,1) },
    'fcc111x11-2'   : {'plane' : (1,1,1), 'x' : (1,1,-2),  'xy' : (-1,1,0) },
    'fcc111x-1-12'  : {'plane' : (1,1,1), 'x' : (-1,-1,2), 'xy' : (1,-1,0) },
    'fcc111x-110'   : {'plane' : (1,1,1), 'x' : (-1,1,0),  'xy' : (-1,-1,2) },
    'fcc111x1-10'   : {'plane' : (1,1,1), 'x' : (1,-1,0),  'xy' : (1,1,-2) },

    'cubic-diamond_primitive' : { },
    'cubic-diamond001x100'    : {'plane' : (0,0,1), 'x' : (1,0,0),   'xy' : (0,1,0) },
    'cubic-diamond110x-110'   : {'plane' : (1,1,0), 'x' : (-1,1,0),  'xy' : (0,0,1) },
    'cubic-diamond111x11-2'   : {'plane' : (1,1,1), 'x' : (1,1,-2),  'xy' : (-1,1,0) },
    'cubic-diamond111x-1-12'  : {'plane' : (1,1,1), 'x' : (-1,-1,2), 'xy' : (1,-1,0) },
    'cubic-diamond111x-110'   : {'plane' : (1,1,1), 'x' : (-1,1,0),  'xy' : (-1,-1,2) },
    'cubic-diamond111x1-10'   : {'plane' : (1,1,1), 'x' : (1,-1,0),  'xy' : (1,1,-2) },
    
    'bcc_primitive' : { },
    'bcc001x100'    : {'plane' : (0,0,1), 'x' : (1,0,0),   'xy' : (0,1,0) },
    'bcc111x-110'   : {'plane' : (1,1,1), 'x' : (-1,1,0),  'xy' : (-1,-1,2) },
    'bcc110x-111'   : {'plane' : (1,1,0), 'x' : (-1,1,1),  'xy' : (0,0,1) },
    'bcc110x1-1-1'  : {'plane' : (1,1,0), 'x' : (1,-1,-1), 'xy' : (0,0,-1) },
    'bcc112x11-1'   : {'plane' : (1,1,2), 'x' : (1,1,-1),  'xy' : (-1,1,0) },
    'bcc112x-1-11'  : {'plane' : (1,1,2), 'x' : (-1,-1,1), 'xy' : (1,-1,0) },
    'bcc123x11-1'   : {'plane' : (1,2,3), 'x' : (1,1,-1),  'xy' : (-2,1,0) },
    'bcc123x-1-11'  : {'plane' : (1,2,3), 'x' : (-1,-1,1), 'xy' : (2,-1,0) },
    
    'hcp_primitive'   : { },
    
    # Basal, cleavage and shear (non SF)
    'hcp0001x2-1-10'  : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,0,0,1]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([2,-1,-1,0]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([0,1,-1,0])
                        },
                        
    # Basal, shear SF1 along x
    'hcp0001x1-100'   : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,0,0,1]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([1,-1,0,0]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([0,1,-1,0])
                        },

    # Basal, shear opposite to SF1 along x, climbing the hill
    'hcp0001x10-10'   : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,0,0,1]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([1,0,-1,0]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([0,1,-1,0])
                        },
    
    # Prism I, cleavage and SF1 along x
    'hcp01-10x-2110'  : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,1,-1,0]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([-2,1,1,0]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([0,0,0,1])
                        },
    
    # Prism I, shear SF2 along x
    'hcp01-10x-2113'  : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,1,-1,0]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([-2,1,1,3]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([0,0,0,1])
                        },       
    
    # Prism II, cleavage
    'hcp-12-10x-1010' : {'plane' : mod_crystal.plane_miller_bravais_to_miller([-1,2,-1,0]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([-1,0,1,0]), 
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([0,0,0,1])
                        },  

    # Pyramidal I, cleavage and <a> slip along x
    'hcp01-11x-2110'  : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,1,-1,1]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([-2,1,1,0]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([1,-2,1,3])
                        },
                        
    # Pyramidal I, <c+a> slip along x
    'hcp01-11x1-213'  : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,1,-1,1]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([1,-2,1,3]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([2,-1,-1,0])
                        },                          
    
    # Pyramidal I, shear SF2 along x
    'hcp01-11x0-112'  : {'plane' : mod_crystal.plane_miller_bravais_to_miller([0,1,-1,1]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([0,-1,1,2]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([1,-2,1,3])
                        },  
    
    # Pyramidal II, Cleavage
    'hcp-12-12x-1010' : {'plane' : mod_crystal.plane_miller_bravais_to_miller([-1,2,-1,2]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([-1,0,1,0]), 
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([1,-2,1,3]) },
    
    # Pyramidal II, shear SF2 along x
    'hcp-12-12x1-213' : {'plane' : mod_crystal.plane_miller_bravais_to_miller([-1,2,-1,2]),
                         'x' : mod_crystal.direction_miller_bravais_to_miller([1,-2,1,3]),
                         'xy' : mod_crystal.direction_miller_bravais_to_miller([1,0,-1,0])
                        }

}

dict_surface_system_names = {

    'fcc001' : 'fcc001x100',
    'fcc110' : 'fcc110x-110',
    'fcc111' : 'fcc111x11-2',

    'cubic-diamond001' : 'cubic-diamond001x100',
    'cubic-diamond110' : 'cubic-diamond110x-110',
    'cubic-diamond111' : 'cubic-diamond111x11-2',
    
    'bcc001' : 'bcc001x100',
    'bcc111' : 'bcc111x-110',
    'bcc110' : 'bcc110x-111',
    'bcc112' : 'bcc112x11-1',
    'bcc123' : 'bcc123x11-1',
    
    'hcpbasal' : 'hcp0001x2-1-10',
    'hcpprism1' : 'hcp01-10x-2110',
    'hcpprism2' : 'hcp-12-10x-1010',
    'hcppyramidal1' : 'hcp01-11x-2110',
    'hcppyramidal2' : 'hcp-12-12x-1010'
}

##################
### SUBROUTINE ###
##################


def set_initial_magnetic_moments(atoms, initial_magmom_per_site):

    if initial_magmom_per_site is not None:
        atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site)
    return atoms

##################
### SUBROUTINE ###
##################


def get_atomic_system(
        atomic_system, lattice, species_list, selective_dynamics=None,
        n_layer=1, min_vacuum_size=0.0, initial_magmom_per_site=None, is_primitive=False, is_ase=False):

    '''
    This subroutine creates a various oriented crystals.
    Inputs:
      atomic_system: Any of the names atomic systems as below such as 'fcc001x100' etc.
      lattice: A pymatgen lattice object with lattice parameters
      species_list: The list of species. For fcc , cubic-diamond, bcc and hcp we just use the first species in the list
    '''
    
    if atomic_system == 'fcc_primitive':
       
        if is_ase:
            atoms = crystal(species_list[0], [(0, 0, 0)], spacegroup=225, cellpar=[lattice.a, lattice.a, lattice.a, 90, 90, 90], primitive_cell=True)
        else:
            structure = mod_pymatgen.get_fcc_standard_structure(symbol=species_list[0], a=lattice.a, is_primitive=True)

    elif atomic_system == 'cubic-diamond_primitive':
       
        if is_ase:
            atoms = bulk(species_list[0], crystalstructure='diamond', a=lattice.a)
        else:
            structure = mod_pymatgen.get_cubic_diamond_standard_structure(symbol=species_list[0], a=lattice.a, is_primitive=True)

    elif atomic_system == 'bcc_primitive':
       
        if is_ase:
            atoms = crystal(species_list[0], [(0, 0, 0)], spacegroup=229, cellpar=[lattice.a, lattice.a, lattice.a, 90, 90, 90], primitive_cell=True)
        else:
            structure = mod_pymatgen.get_bcc_standard_structure(symbol=species_list[0], a=lattice.a, is_primitive=True)
        
    elif atomic_system == 'hcp_primitive' or atomic_system == 'hcp_conventional':
       
        if is_ase:
            atoms = bulk(species_list[0], crystalstructure='hcp', a=lattice.a, c=lattice.c)
        else:
            structure = mod_pymatgen.get_hcp_standard_structure(symbol=species_list[0], a=lattice.a, c=lattice.c, is_primitive=True)
        
    elif atomic_system == 'fcc_conventional':
       
        if is_ase:
            directions = [[1,0,0],[0,1,0],[0,0,1]]
            atoms = fcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        else:
            structure = mod_pymatgen.get_fcc_standard_structure(symbol=species_list[0], a=lattice.a)

    elif atomic_system == 'cubic-diamond_conventional':
       
        if is_ase:
            directions = [[1,0,0],[0,1,0],[0,0,1]]
            atoms = Diamond(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        else:
            structure = mod_pymatgen.get_cubic_diamond_standard_structure(symbol=species_list[0], a=lattice.a)

    elif atomic_system == 'bcc_conventional':
       
        if is_ase:
            directions = [[1,0,0],[0,1,0],[0,0,1]]
            atoms = bcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        else:
            structure = mod_pymatgen.get_bcc_standard_structure(symbol=species_list[0], a=lattice.a)
        
    else:
    
        if 'fcc' in atomic_system:
            conv_structure = mod_pymatgen.get_fcc_standard_structure(symbol=species_list[0], a=lattice.a)
        if 'cubic-diamond' in atomic_system:
            conv_structure = mod_pymatgen.get_cubic_diamond_standard_structure(symbol=species_list[0], a=lattice.a)
        elif 'bcc' in atomic_system:
            conv_structure = mod_pymatgen.get_bcc_standard_structure(symbol=species_list[0], a=lattice.a)
        elif 'hcp' in atomic_system:
            conv_structure = mod_pymatgen.get_hcp_standard_structure(symbol=species_list[0], a=lattice.a, c=lattice.c)  
            # NOTE: hcp 1 layer to for Prism II plane gives a left handed system, so use 2 layers instead, anyways the system will be reduced to primitive
            if n_layer == 1:
                n_layer = 2
     
        structure = mod_pymatgen.create_oriented_slab(
            structure=conv_structure,
            plane_miller_index=dict_atomic_system[atomic_system]['plane'],
            n_layer=n_layer,
            min_vacuum_size=min_vacuum_size,
            is_reorient_slab=True,
            x_miller_index=dict_atomic_system[atomic_system]['x'],
            xy_miller_index=dict_atomic_system[atomic_system]['xy'],
            is_primitive=is_primitive)
    
    if not is_ase:        
        atoms = AseAtomsAdaptor.get_atoms(structure)

    if initial_magmom_per_site is not None:
        atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site)      
        
    return atoms
    
##################
### SUBROUTINE ###
##################

def get_surface_system_pymatgen_derived(
    surface_system, species, lattice, n_layer, min_vacuum_size,
    initial_magmom_per_site, is_vca=False, secondary_species=None):

    '''
      Setup a surface energy calculation
    '''
    
    # Get the full name of the atomic system
    atomic_system = dict_surface_system_names[surface_system]
    
    # Create the atomic system
    atoms = get_atomic_system(
        atomic_system=atomic_system,
        lattice=lattice,
        species_list = [species],
        selective_dynamics=None,
        n_layer=n_layer, min_vacuum_size=min_vacuum_size,
        initial_magmom_per_site=initial_magmom_per_site,
        is_primitive=True)
        
    # Add constraints of (FFT) to each atom
    mod_ase.add_constraints(atoms, selective_dynamics=(1,1,0))
        
    # Add vca atoms
    if is_vca:
        atoms = mod_ase.vca_superpose_secondary_ase_atoms(in_atoms=atoms, secondary_species=secondary_species)
    
    return atoms
    
##################
### SUBROUTINE ###
##################

def get_surface_system_pymatgen_inbuilt(
        structure_type, lattice, species, surface_miller_index, min_slab_size,
        min_vacuum_size, initial_magmom_per_site=None, is_vca=False,
        secondary_species=None, max_index=2, is_primitive=True, selective_dynamics=None):
        
    '''
      Given a miller index of a plane, this subroutine setsup the surface
      systems with all possible terminations.
    '''
    
    mod_utility.error_check_argument_required(
        arg_val=structure_type, arg_name='structure_type', module=module_name,
        subroutine='get_surface_system_pymatgen_inbuilt',
        valid_args=['fcc', 'bcc', 'hcp'])

    # Get conventional structure
    if structure_type == 'fcc':
        structure = mod_pymatgen.get_fcc_standard_structure(symbol=species, a=lattice.a)
        
    elif structure_type == 'bcc':
        structure = mod_pymatgen.get_bcc_standard_structure(symbol=species, a=lattice.a)
        
    elif structure_type == 'hcp':
        structure = mod_pymatgen.get_hcp_standard_structure(symbol=species, a=lattice.a, c=lattice.c)

    all_slabs = generate_all_slabs(
        structure=structure,
        max_index=max_index,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        bonds=None,
        tol=1e-3,
        max_broken_bonds=0,
        lll_reduce=False,
        center_slab=True,
        primitive=is_primitive,
        max_normal_search=2,
        symmetrize=True,
        repair=True)
    
    # There can be multiple terminations for a given miller index plane
    selected_slabs = []
    for slab in all_slabs:
        if slab.miller_index == surface_miller_index:
            atoms = AseAtomsAdaptor.get_atoms(slab)
            if initial_magmom_per_site is not None:
                atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site) 
            selected_slabs.append( atoms )
            
    if is_vca:
        for sidx in range(0,len(selected_slabs)):
            selected_slabs[sidx] = mod_ase.vca_superpose_secondary_ase_atoms(in_atoms=selected_slabs[sidx], secondary_species=secondary_species)
            
    if selective_dynamics is not None:
        for sidx in range(0,len(selected_slabs)):
            mod_ase.add_constraints(selected_slabs[sidx], selective_dynamics=selective_dynamics)
            
    return selected_slabs

##################
### SUBROUTINE ###
##################


def get_atomic_system_ase(
        atomic_system, lattice, species_list, selective_dynamics=None,
        n_layer=None, initial_magmom_per_site=None):

    '''
    This subroutine creates a various oriented crystals.
    Inputs:
      atomic_system: Any of the names atomic systems as below such as 'fcc001x100' etc.
      lattice: A pymatgen lattice object with lattice parameters
      species_list: The list of species. For fcc , bcc and hcp we just use the first species in the list
    '''

    rotate_tag = ''
    #######################
    #         FCC         #
    #######################
    if atomic_system == 'fcc_primitive':
    
        # Description: FCC primitive cell
        # Used for hydrostatic loading and volume relaxation for lattice parameters   
    
        atoms = crystal(species_list[0], [(0, 0, 0)], spacegroup=225, cellpar=[lattice.a, lattice.a, lattice.a, 90, 90, 90], primitive_cell=True)

        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(1) * initial_magmom_per_site )
        return atoms
        
    elif atomic_system == 'fcc001x100':

        # Description: FCC (001) surface with x axis along [100], y axis in (001) plane
        # Used for uniaxial stress and surface decohesion

        #directions = [[1,0,0],[0,1,0],[0,0,1]]  # 4 atoms per unit cell, V = 4V0
        #directions = [[1,0,0],[1,1,0],[0,0,1]]  # 2 atoms per unit cell, V = 2V0
        directions = [[1,0,0],[1,1,0],[1,0,1]]   # 1 atom per unit cell,  V = V0

        atoms = fcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms
        
    elif atomic_system == 'fcc110x-110':

        # Description: FCC (110) surface with x axis along [-110], y axis in (110) plane
        # Used for uniaxial stress and surface decohesion

        #directions = [[-1,1,0], [0, 0, 1], [1,1,0]] # 2 atoms per unit cell, V = 2V0
        directions = [[-1,1,0], [-1,1,2], [0,1,1]]   # 1 atoms per unit cell, V = V0
        
        atoms = fcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms             

    elif atomic_system == 'fcc111x11-2' or atomic_system == 'fcc111x-1-12': 

        # Description:
        #   fcc111x11-2: (111) surface with x axis long
        # Used for uniaxial stress or decohesion and stacking faults
        # NOTE: If the slip along x is reversed, it is NOT symmetric
        
        if atomic_system == 'fcc111x11-2':
            #directions = [[1,1,-2], [-1,1,0], [1,1,1]] # 6 atoms per unit cell, V = 6V0
            #directions = [[1,1,-2], [0,1,-1], [1,1,1]] # 3 atoms per unit cell, V = 3V0
            directions = [[1,1,-2], [0,1,-1], [1,1,0]]  # 1 atom  per unit cell, V = V0
 
        elif atomic_system == 'fcc111x-1-12':
            #directions = [[-1,-1,2], [1,-1,0], [1,1,1]] # 6 atoms per unit cell, V = 6V0
            #directions = [[-1,-1,2], [0,-1,1], [1,1,1]] # 3 atoms per unit cell, V = 3V0
            directions = [[-1,-1,2], [0,-1,1], [1,1,0]] # 1 atom per unit cell, V = V0
        
        atoms = fcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms
        
    elif atomic_system == 'fcc111x-110' or atomic_system == 'fcc111x1-10': 

        # Used for uniaxial stress or decohesion and stacking faults
        # NOTE: This is similar to the above 'fcc111x11-2' or 'fcc111x-1-12' but slip is in the orthogonal direction
        # NOTE: If the slip along x is reversed, it is symmetric
        
        if atomic_system == 'fcc111x-110':
            #directions = [[-1,1,0], [-1,-1,2], [1,1,1]] # 6 atoms per unit cell, V = 6V0
            #directions = [[-1,1,0], [-1,0,1], [1,1,1]]  # 3 atoms per unit cell, V = 3V0
            directions = [[-1,1,0], [-1,0,1], [0,1,1]]   # 1 atom per unit cell, V= V0
 
        elif atomic_system == 'fcc111x1-10':
            #directions = [[1,-1,0], [1,1,-2], [1,1,1]]  # 6 atoms per unit cell, V = 6V0
            #directions = [[1,-1,0], [1,0,-1], [1,1,1]]  # 3 atoms per unit cell, V = 3V0
            directions = [[1,-1,0], [1,0,-1], [0,1,1]]   # 1 atom per unit cell, V= V0
        
        atoms = fcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms        

    #######################
    #         BCC         #
    #######################
    # For reference: Specified volumes are for Fe with a = 2.830079278000
    elif atomic_system == 'bcc_primitive':
    
        atoms = crystal(species_list[0], [(0, 0, 0)], spacegroup=229, cellpar=[lattice.a, lattice.a, lattice.a, 90, 90, 90], primitive_cell=True)

        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(1) * initial_magmom_per_site )
        return atoms

    elif atomic_system == 'bcc001x100':

        # Used for uniaxial stress or decohesion

        #directions = [[1,0,0],[0,1,0],[0,0,1]] # 2 atoms, V = 22.67
        directions = [[1,0,0],[1,1,0],[1,1,1]]  # 1 atom, V = 11.33

        atoms = bcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms
        
    elif atomic_system == 'bcc111x-110':

        # Used for uniaxial stress or decohesion
     
        #directions = [[-1,1,0],[-1,-1,2],[1,1,1]] # 6 atoms, V = 68.00
        #directions = [[-1,1,0],[-1,0,1],[1,1,1]]  # 3 atoms, V = 34.00
        directions = [[-1,1,0],[-1,0,1],[-1,1,1]]  # 1 atom, V = 11.33

        atoms = bcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )

        return atoms
       
    elif atomic_system == 'bcc110x-111' or atomic_system == 'bcc110x1-1-1':
    
        # Used for BCC uniaxial as well as slip system of (110) plane and [-111] slip direction (symmetric)

        if atomic_system == 'bcc110x-111':
            #directions = [[-1,1,1],[0,0,1],[1,1,0]] # 2 atoms, V = 22.68
            directions = [[-1,1,1],[0,0,1],[1,1,1]] # 1 atoms, V = 11.33
            
        elif atomic_system == 'bcc110x1-1-1':
            #directions = [[1,-1,-1],[0,0,-1],[1,1,0]]  # 2 atoms, V = 22.68
            directions = [[1,-1,-1],[0,0,-1],[1,1,1]] # 1 atoms, V = 11.33

        atoms = bcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms         
        
    elif atomic_system == 'bcc112x11-1' or atomic_system == 'bcc112x-1-11':
    
        # Used for BCC slip system of (112) plane and [11-1] slip direction
        
        # Hard direction of slip
        if atomic_system == 'bcc112x11-1':
            #directions = [[1,1,-1],[-1,1,0],[1,1,2]]  # 6 atoms per unit cell, V =68.00
            #directions = [[1,1,-1],[-1,1,0],[1,1,1]]  # 2 atoms per unit cell, V =34.00
            directions = [[1,1,-1],[-1,1,0],[0,1,0]]   # 1 atom, V = 11.33
            # Your reciprocal lattice and k-lattice belong to different lattice classes. Although this is a warning, code breaks after few loadings
            
        # Easy direction of slip
        elif atomic_system == 'bcc112x-1-11':
            #directions = [[-1,-1,1],[1,-1,0],[1,1,2]] # 6 atoms per unit cell
            #directions = [[-1,-1,1],[1,-1,0],[1,1,1]]  # 2 atoms per unit cell
            directions = [[-1,-1,1],[1,-1,0],[0,1,0]] # 1 atom per unit cell

        atoms = bcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms        
        
    elif atomic_system == 'bcc123x11-1' or atomic_system == 'bcc123x-1-11':
    
        # Used for BCC slip system of (123) plane and [11-1] slip direction
        # We create a non orthogonal cell with first or x axis along [11-1], second in-plane axis along [-210] and resultant thrid or z axis along [123]
        
        # Easy direction of slip
        if atomic_system == 'bcc123x11-1':
            #directions = [[1,1,-1],[-2,1,0],[1,2,3]]  # 14 atoms per unit cell, V = 158.67
            #directions = [[1,1,-1],[-2,1,0],[-2,1,2]] # 6 atoms per unit cell, V = 68.00, no warning
            #directions = [[1,1,-1],[-2,1,0],[0,0,1]]  # 3 atoms per unit cell, V = 34.00, but k-point warning
            directions = [[1,1,-1],[-2,1,0],[0,1,0]]   # 2 atoms per unit cell, V = 22.67, but k-point warning
            
        # Hard direction of slip
        elif atomic_system == 'bcc123x-1-11':
            #directions = [[-1,-1,1],[2,-1,0],[1,2,3]]  # 14 atoms per unit cell
            #directions = [[-1,-1,1],[2,-1,0],[-2,1,2]] # 6 atoms per unit cell, no warning            
            #directions = [[-1,-1,1],[2,-1,0],[0,0,1]]  # 3 atoms per unit cell, but k-point warning
            directions = [[-1,-1,1],[2,-1,0],[0,1,0]]   # 2 atoms per unit cell, V = 22.67, but k-point warning            


        atoms = bcc(directions=directions, size=(1,1,1), symbol=species_list[0], latticeconstant=lattice.a)
        
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )
           
        return atoms

    #######################
    #         HCP         #
    #######################
    # References: 1. Comprehensive ﬁrst-principles study of stable stacking faults in hcp metals, Yin, Wu, Curtin 2017
    #             2. Local deformation mechanisms of two-phase Ti alloy, Jun, 2016
    
    elif atomic_system == 'hcp_primitive':
    
        atoms = bulk(species_list[0], crystalstructure='hcp', a=lattice.a, c=lattice.c)

        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )            
        return atoms
        
    elif (atomic_system == 'basal') or (atomic_system == 'hcp0001x2-1-10') or (atomic_system == 'hcp0001x10-10'):

        # Description: HCP (0001) basal plane
        #     'hcp0001x2-1-10': x direction is [2-1-10] is fill slip
        #                      cell vectors a1 is orth. a3, a2 is orth. a3 but a1 not orth. a2
        #     'hcp0001x10-10': x direction is [10-10] is partial slip
        #                      Instead of rotation it -90 degrees we rotate it -30 degrees about z axis as [01-10] is equivalent to [10-10]
        
        directions = [[2,-1,-1,0], [-1,2,-1,0], [0,0,0,1]] # 2 atoms per unit cell
        
        atoms = hcp(directions=directions,size=(1,1,1),symbol=species_list[0],latticeconstant={'a': lattice.a, 'c': lattice.c})

        if atomic_system == 'hcp0001x10-10':
            # Angle between [2-1-10] and [10-10]
            atoms.rotate(-30, 'z', rotate_cell=True)
            
        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )            
            
        return atoms

    elif (atomic_system == 'prism1') or \
         (atomic_system == 'hcp01-10x-2110') or (atomic_system == 'hcp01-10x-2110w') or (atomic_system == 'prism1xSF1') or \
         (atomic_system == 'prism1xSF2'):

        # Description: HCP (0001) basal plane
        #     'hcp01-10x-2110' or 'hcp01-10x-2110w' or 'prism2xSF1': x direction is [-2110] leading to SF1
        #        NOTE: This is an orthogonal cell
        #     'prism2xSF2': x direction is along the diagonal of [-2,1,1,0] and [0,0,0,1], along which SF2 is located
        
        directions = [[-2,1,1,0], [0,0,0,1], [0,1,-1,0]] # 4 atoms per unit cell
        
        atoms = hcp(directions=directions,size=(1,1,1),symbol=species_list[0],latticeconstant={'a': lattice.a, 'c': lattice.c})

        if atomic_system == 'prism1xSF2':
            # theta = math.degrees( math.atan2( lattice.c , lattice.a ) ) # This is same as below
            # In this case a1 and a2 are orthogonal, |a1| = a, |a2| = c
            cellpar_list  = atoms.cell.cellpar()
            a1 = cellpar_list[0] # Equal to a
            a2 = cellpar_list[1] # Equal to c

            theta = math.degrees( math.atan2( a2 , a1 ) )
            atoms.rotate(-1.0 * theta, 'z', rotate_cell=True)

        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )            
            
        return atoms
       
    elif (atomic_system == 'prism2') or (atomic_system == 'hcp-12-10x-1010'):

        directions = [[-1,1,0,0], [0,0,0,1], [1,1,-2,0]]
        
        atoms = hcp(directions=directions,size=(1,1,1),symbol=species_list[0],latticeconstant={'a': lattice.a, 'c': lattice.c})        

        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(len(atoms.positions)) * initial_magmom_per_site )            
            
        return atoms

    elif (atomic_system == 'pyramidal-1') or (atomic_system == 'hcp01-11x2-1-10') or (atomic_system == 'hcp01-11x2-1-10w') or (atomic_system == 'hcp01-11x0-112w'):

        # Pyramidal I: 
        c1 = '[-2110]'
        c2 = '[1-213]'
        c3 = '[-12-10]'

        system_tags = 'hcp ' + str(lattice.a) + ' ' + str(lattice.c)

        if atomic_system == 'hcp01-11x0-112w':

            # Angle between [-2110] and [0-112] on the slip plane
            rotate_tag = '-rotate z -90'

    elif (atomic_system == 'pyramidal-2') or (atomic_system == 'hcp-12-12x-1010') or (atomic_system == 'hcp-12-12x1-213'):

        # Pyramidal II:
        c1 = '[-1010]'
        c2 = '[1-213]'
        c3 = '[-12-10]'
        system_tags = 'hcp ' + str(lattice.a) + ' ' + str(lattice.c)

        if atomic_system == 'hcp-12-12x1-213':

            # Angle between [-1010] and [1-213] on the slip plane
            rotate_tag = '-rotate z -90'

    else:

        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'get_atomic_system'\n")
        sys.stderr.write("       Unknown value for argument 'atomic_system'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    #------------------------------------------------
    # Create basic layer to count atoms in the layer
    #------------------------------------------------
    if os.path.isfile('POSCAR'):
        os.remove('POSCAR')
    atomsk_command_basic = 'atomsk --create ' + system_tags + ' ' +  \
        ' '.join([species_list[0]]) + '  orient ' + c1 + ' ' + c2 + ' ' + c3 + ' ' \
        + rotate_tag + ' POSCAR'
    process = subprocess.call(atomsk_command_basic, shell=True)
    atoms_basic = read_vasp('POSCAR')
    os.remove('POSCAR')
    
    # Count number of layers in the z direction
    [basic_layer_idx_arr, basic_z_arr] = geometry.get_layers(atoms_basic, [0,0,1])
    n_layer_basic = len(basic_z_arr)

    #------------------
    # Create supercell
    #------------------
    atoms_supercell_list = [None, None]
    for i in range(0, len(species_list)):

        if n_layer is not None:
            n_z_periodic = math.ceil(n_layer / n_layer_basic)
            atomsk_command_supercell = 'atomsk --create ' + system_tags + ' ' +  \
                ' '.join([species_list[i]]) + '  orient ' + c1 + ' ' + c2 + ' ' + c3 + \
                ' -duplicate 1 1 ' + str(n_z_periodic) + ' ' + rotate_tag + ' POSCAR'
        else:
            atomsk_command_supercell = 'atomsk --create ' + system_tags + ' ' +  \
                ' '.join([species_list[i]]) + '  orient ' + c1 + ' ' + c2 + ' ' + c3 + ' ' \
                + rotate_tag + ' POSCAR'

        process = subprocess.call(atomsk_command_supercell, shell=True)
        atoms_supercell_list[i] = read_vasp('POSCAR')
        os.remove('POSCAR')

        if selective_dynamics is not None:
            atoms_supercell_list[i] = add_constraints(atoms_supercell_list[i], selective_dynamics)
        
    
    '''
    #---------------------------
    # Delete layers if required
    #---------------------------
    [super_layer_idx_arr, super_z_arr] = geometry.get_layers(atoms_supercell, [0,0,10])

    # If layers are equispaced, just delete the extra layers
    if atomic_system in ['fcc001x100', 'fcc110x-110', 'fcc111x11-2', 'fcc111x-110', \
                       'bcc001x100', 'bcc111x11-2', 'bcc111x-110', 'bcc110x-110', 'bcc110x-111', 'bcc211x-111', 'bcc321x-111', \
                       'basal', 'hcp0001x2-1-10', 'hcp0001x01-10', \
                       'prism-2', 'hcp-12-10x-1010', \
                       'pyramidal-2', 'hcp-12-12x-1010', 'hcp-12-12x1-213']:

        del atoms_supercell[[atom.index for atom in atoms_supercell if (super_layer_idx_arr[atom.index] >= n_layer)]]
        cell = atoms_supercell.get_cell()
        layer_height = cell[2][2] / len(super_z_arr)
        cell[2][2] = n_layer * layer_height
        atoms_supercell.set_cell(cell)

    # Prism I plane in HCP has two spacings (Narrow and Wide), we are usually interested in wide spacing
    elif atomic_system in ['prism-1', 'hcp01-10x-2110', 'hcp01-10x-2110w']:

    # Pyramidal I plane in HCP has two spacings (Narrow and Wide), we are usually interested in wide spacing
    elif atomic_system in ['pyramidal-1', 'hcp01-11x2-1-10', 'hcp01-11x2-1-10w', 'hcp01-11x0-112w']:

    '''

    if initial_magmom_per_site is not None:
        atoms_supercell_list[0].set_initial_magnetic_moments( np.ones(len(atoms_supercell_list[0].positions)) * initial_magmom_per_site )

    return atoms_supercell_list[0]



    
    
'''

    elif (atomic_system == 'prism-1') or (atomic_system == 'hcp01-10x-2110') or (atomic_system == 'hcp01-10x-2110w'):

        # Prism I: c1 orth. c2 orth. c3

        c1 = '[-2110]'
        c2 = '[0001]'
        c3 = '[01-10]'
        system_tags = 'hcp ' + str(lattice.a) + ' ' + str(lattice.c)

    elif (atomic_system == 'basal') or (atomic_system == 'hcp0001x2-1-10') or (atomic_system == 'hcp0001x01-10'):

        # Basal: c1 orth. c3, c2 orth. c3 but c1 not orth. c2
        # Shear along x is full slip
        # Shear along y is partial slip

        c1 = '[2-1-10]'
        c2 = '[-12-10]'
        c3 = '[0001]'
        system_tags = 'hcp ' + str(lattice.a) + ' ' + str(lattice.c)

        if atomic_system == 'hcp0001x01-10':

            # Angle between [2-1-10] and [01-10]
            rotate_tag = '-rotate z -90'



    elif atomic_system == 'hcp_primitive':
    
        atoms = bulk(species_list[0], crystalstructure='hcp', a=lattice.a, c=lattice.c)

        if initial_magmom_per_site is not None:
            atoms.set_initial_magnetic_moments( np.ones(1) * initial_magmom_per_site )            
        return atoms
        
        
        
        

    elif atomic_system == 'fcc111x-110':  

        # Used for uniaxial stress or decohesion and stacking faults
        # NOTE: Symmetric even if slip along x is reversed

        c1 = '[-110]'
        c2 = '[-1-12]'
        c3 = '[111]'
        system_tags = 'fcc ' + str(lattice.a)

    elif atomic_system == 'fcc111x-1-12': 

        # Used for uniaxial stress or decohesion and stacking faults
        # NOTE: See above case, opposite slip to above

        c1 = '[-1-12]'
        c2 = '[1-10]'
        c3 = '[111]'
        system_tags = 'fcc ' + str(lattice.a)

    elif atomic_system == 'fcc111x11-2': 

        # Used for uniaxial stress or decohesion and stacking faults
        # NOTE: If the slip along x is reversed, NOT symmetric
        # NOTE: We can manually create 1 atom per layer cell but atomsk needs
        #       orthogonal cell for cubic systems. So this setup has 6 atoms in 3 layers

        c1 = '[11-2]'
        c2 = '[-110]'
        c3 = '[111]'
        system_tags = 'fcc ' + str(lattice.a)


        
    elif atomic_system == 'fcc110x-110':

        # Used for uniaxial stress or decohesion

        c1 = '[-110]'
        c2 = '[001]'
        c3 = '[110]'
        system_tags = 'fcc ' + str(lattice.a)

    elif atomic_system == 'fcc001x100':

        # Used for uniaxial stress or decohesion

        c1 = '[100]'
        c2 = '[010]'
        c3 = '[001]'
        system_tags = 'fcc ' + str(lattice.a)
        
        

    elif atomic_system == 'bcc111x11-2':

        # Used for uniaxial stress or decohesion

        c1 = '[11-2]'
        c2 = '[-110]'
        c3 = '[111]'
        system_tags = 'bcc ' + str(lattice.a)

    elif atomic_system == 'bcc111x-110':

        # Used for uniaxial stress or decohesion (same as bcc111x11-2)

        c1 = '[-110]'
        c2 = '[-1-12]'
        c3 = '[111]'
        system_tags = 'bcc ' + str(lattice.a)


    elif (atomic_system == 'bcc110x-110') or (atomic_system == 'bcc110x-111'):

        # Used for uniaxial stress or decohesion and stacking faults
        # NOTE:  'bcc110x1-12' is same as 'bcc110x-111'
        # NOTE: All three are symmetric w.r.t. direction of slip

        c1 = '[-110]'
        c2 = '[001]'
        c3 = '[110]'
        system_tags = 'bcc ' + str(lattice.a)
        if atomic_system == 'bcc110x-111':

            # Angle between [-110] and [-111]
            theta = -1.0*math.degrees(math.atan(1/math.sqrt(2)))
            rotate_tag = '-rotate z ' + str(theta)


    elif atomic_system == 'bcc211x-111':

        # Used for uniaxial stress or decohesion and stacking faults

        c1 = '[-111]'
        c2 = '[0-11]'
        c3 = '[211]'
        system_tags = 'bcc ' + str(lattice.a)

    elif atomic_system == 'bcc321x-111':

        # Used for uniaxial stress or decohesion and stacking faults

        c1 = '[-111]'
        c2 = '[1-45]'
        c3 = '[321]'
        system_tags = 'bcc ' + str(lattice.a)
'''        

'''----------------------------------------------------------------------------
                              END OF MODULE
----------------------------------------------------------------------------'''
