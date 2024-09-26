'''----------------------------------------------------------------------------
                                mod_pymatgen.py

 Description: General routines related to pymatgen.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import numpy as np

# Externally installed modules
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.core.surface import SlabGenerator, generate_all_slabs

# Local imports
from matkit.core import mod_math

'''----------------------------------------------------------------------------
                                MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_pymatgen.py"

'''----------------------------------------------------------------------------
                                    SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################


def get_pymatgen_results_dict(contcar, outcar):

    results = {
        "energy_eV" : np.array([]),
        "stress_voigt_GPa" : np.empty((0,6)),
        "volume_A3" : np.array([]),
        "cellpar_A" : np.empty((0,6)),
        "cellvecs_A" : None,
        "magmoms" : np.array([]),
        "forces" : np.array([]),
        "n_atoms" : 0
    }

    results["energy_eV"] = outcar.final_energy

    results["stress_voigt_GPa"] = in_atoms.get_stress() * \
        mod_units.eV_per_cubic_Angstrom_to_Pa * (1.0E-9)

    results["volume_A3"] = in_atoms.calc.atoms.get_volume()

    results["cellpar_A"] = \
        geometry.cell.cell_to_cellpar(in_atoms.calc.atoms.get_cell())

    results["cellvecs_A"] = in_atoms.get_cell()[:]

    magmom_list = []
    for magmom in outcar.magnetization:
        magmom_list.append(magmom['tot'])
    results["magmoms"] = magmom_list

    results["forces"] = in_atoms.get_forces()

    results["n_atoms"] = len(in_atoms.get_chemical_symbols())

    return results

##################
### SUBROUTINE ###
##################
def print_dft_results_from_pymatgen_results(contcar, outcar, filename="dft_results.json"):

    dft_results = get_pymatgen_results_dict(contcar, outcar)

    with open(filename, "w") as fh:
        json.dump(dft_results, fh, indent=4,
                  cls=mod_utility.json_numpy_encoder)

##################
### SUBROUTINE ###
##################

def get_fcc_standard_structure(symbol, a, is_primitive=False):

    lattice = Lattice.cubic(a=a)
    structure = Structure(lattice, [symbol] * 4 , [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])

    return get_standard_structure(structure, is_primitive)
    
##################
### SUBROUTINE ###
##################

def get_cubic_diamond_standard_structure(symbol, a, is_primitive=False):

    lattice = Lattice.cubic(a=a)
    structure = Structure(lattice, [symbol] * 8 , [[0, 0, 0], [1/4, 1/4, 1/4], [1/2, 1/2, 0], [3/4, 3/4, 1/4], [1/2, 0, 1/2], [3/4, 1/4, 3/4], [0, 1/2, 1/2], [1/4, 3/4, 3/4]])
    
    return get_standard_structure(structure, is_primitive)
    
##################
### SUBROUTINE ###
##################

def get_bcc_standard_structure(symbol, a, is_primitive=False):

    lattice = Lattice.cubic(a=a)
    structure = Structure(lattice, [symbol] * 2 , [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return get_standard_structure(structure, is_primitive)
    
##################
### SUBROUTINE ###
##################

def get_hcp_standard_structure(symbol, a, c, is_primitive=False):

    lattice = Lattice.hexagonal(a=a, c=c)
    structure = Structure(lattice, [symbol] * 2, [[2/3, 1/3, 0.75], [1/3,2/3,0.25]])
    return get_standard_structure(structure, is_primitive)

##################
### SUBROUTINE ###
##################

def get_standard_structure(structure, is_primitive=False):

    spg_obj = SpacegroupAnalyzer(structure)
    
    if is_primitive:
        return spg_obj.find_primitive()
    else:
        return spg_obj.get_conventional_standard_structure()
        
##################
### SUBROUTINE ###
##################

def get_standard_structure_from_structure_type(structure_type, lattice, is_primitive=False):

    # Get conventional structure
    if structure_type == 'fcc':
        structure = get_fcc_standard_structure(symbol=species, a=lattice.a, is_primitive=is_primitive)

    elif structure_type == 'cubic-diamond':
        structure = get_cubic_diamond_standard_structure(symbol=species, a=lattice.a, is_primitive=is_primitive)
       
    elif structure_type == 'bcc':
        structure = get_bcc_standard_structure(symbol=species, a=lattice.a, is_primitive=is_primitive)
        
    elif structure_type == 'hcp':
        structure = get_hcp_standard_structure(symbol=species, a=lattice.a, c=lattice.c, is_primitive=is_primitive)    

    return structure

##################
### SUBROUTINE ###
##################


def get_interplanar_spacing(structure, plane_miller_index):

    '''
    Finds the interplanar spacing of the conventional structure of the given structure
    Input Arguments:
        structure: The input structure
        plane: The 3 index miller plane (hkl)
    '''
   
    conv_std_structure = get_standard_structure(structure)
    tem_calc_obj = TEMCalculator()
    spacing_dict = tem_calc_obj.get_interplanar_spacings(conv_std_structure, [plane_miller_index])

    return spacing_dict[plane_miller_index]

##################
### SUBROUTINE ###
##################

def create_oriented_slab(
       structure, plane_miller_index, n_layer, min_vacuum_size,
       is_reorient_slab=False, x_miller_index=None, xy_miller_index=None,
       is_lll_reduce=True, is_primitive=False,  is_center_slab=True):

    '''
    Creates a slab structure with the given plane (3 index miller plane (hkl))
    '''
    
    # Slab size
    interplanar_spacing = get_interplanar_spacing(structure, plane_miller_index)
    slab_size = interplanar_spacing * n_layer
    
    # Generate the slab
    conv_std_structure = get_standard_structure(structure)
    slab_gen_obj = SlabGenerator(
        initial_structure=conv_std_structure,
        miller_index=plane_miller_index,
        min_slab_size=slab_size,
        min_vacuum_size=min_vacuum_size,
        reorient_lattice=False, # Do not change this, leave it as false
        lll_reduce=is_lll_reduce,
        primitive=is_primitive,
        center_slab=is_center_slab)

    slabs = slab_gen_obj.get_slabs(ftol = 0.001)           
    slab = [s for s in slabs if s.miller_index == plane_miller_index][0]
   
    # If is_reorient_slab is True, reorient the slab such that x direction is along x_miller_index and z direction is along the normal to the slab
    if is_reorient_slab:
        if x_miller_index is None or xy_miller_index is None:
            sys.stderr.write("Error: In module %s\n" %(module_name))
            sys.stderr.write("       In subroutine 'create_oriented_slab'\n")
            sys.stderr.write("       Both x_miller_index and xy_miller_index must be provided\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)
        else:
           
            # Express x_miller_index and xy_miller_index in the conventional standard cartesian coordinate system
            x_cartesian = np.zeros(3)
            xy_cartesian = np.zeros(3)         
            for idx in range(0,3):
               x_cartesian = x_cartesian + x_miller_index[idx] * conv_std_structure.lattice.matrix[idx]
               xy_cartesian = xy_cartesian + xy_miller_index[idx] * conv_std_structure.lattice.matrix[idx]

            x_cartesian_unit_vector = mod_math.l2_normalize_1d_np_vec(x_cartesian)
            z_cartesian_unit_vector = mod_math.l2_normalize_1d_np_vec( np.cross(x_cartesian, xy_cartesian) )
            y_cartesian_unit_vector = mod_math.l2_normalize_1d_np_vec( np.cross(z_cartesian_unit_vector, x_cartesian_unit_vector) )            
            reoriented_basis = np.array([ x_cartesian_unit_vector, y_cartesian_unit_vector, z_cartesian_unit_vector ])
          
            # Transform the lattice vectors of the slab
            reoriented_lattice_vectors = []
            for idx in range(0,3):
                reoriented_lattice_vectors.append(mod_math.transform_3d_vector(v=slab.lattice.matrix[idx], dest=reoriented_basis))
                
            slab = Structure(lattice=np.matrix(reoriented_lattice_vectors), coords=slab.frac_coords, species=slab.species)
            
    # Order the atoms in the lattice in the increasing order of the third lattice direction
    n_atoms_slab = len(slab.frac_coords)
    order = zip(slab.frac_coords, slab.species)
    c_order = sorted(order, key=lambda x: x[0][2])
    sorted_frac_coords = []
    sorted_species = []
    for (frac_coord, species) in c_order:
        sorted_frac_coords.append(frac_coord)
        sorted_species.append(species)
    slab_lattice = slab.lattice
    slab = Structure(lattice=slab_lattice, coords=sorted_frac_coords, species=sorted_species)
    
    # Slab area
    slab_area = np.linalg.norm(np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1]) )       

    return slab
    
'''----------------------------------------------------------------------------
                                   END OF MODULE
----------------------------------------------------------------------------'''
