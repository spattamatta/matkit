'''----------------------------------------------------------------------------
                                mod_ase.py

 Description: General routines related to ase.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json
import numpy as np
from copy import deepcopy
from shutil import copyfile
from scipy.interpolate import interp1d

# Externally installed modules
from ase.build import sort
from ase import Atom, Atoms, geometry
from ase.constraints import FixAtoms, FixedLine, FixedMode, FixScaled, FixedPlane

# Local imports
from matkit.core import mod_tensor, mod_utility, mod_units

'''----------------------------------------------------------------------------
                                MODULE VATIABLES
----------------------------------------------------------------------------'''
module_name = "mod_ase.py"

'''----------------------------------------------------------------------------
                                    SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################

def add_constraints(atoms, selective_dynamics, indices=None):

    '''
        selective_dynamics is a triple (i, j, k) i,j,k are either 0 or 1
        0 if the degree of freedom is free
        1 if the degree of freedom is fixed
    '''

    if indices is None:
        indices = range(0, len(atoms))

    constraints = []
    for idx in indices:
      constraints.append(FixScaled(atoms.cell, idx, selective_dynamics))
    atoms.set_constraint(constraints)
    
    #for idx in indices:
    #  constraints.append(FixedPlane(indices=idx, direction=selective_dynamics))
    #atoms.set_constraint(constraints)

##################
### SUBROUTINE ###
##################

def vca_superpose_secondary_ase_atoms(in_atoms, secondary_species):

    '''
    Given an atoms object, this subroutine creates VCA atoms, i.e superposed
    secondary_species atoms onto the given atoms
    '''
    
    atoms = deepcopy(in_atoms)
    atoms_secondary = deepcopy(in_atoms)
    
    # Replace the species of the seconday atoms
    atoms_secondary.set_chemical_symbols( [secondary_species] * in_atoms.get_global_number_of_atoms() )
    atoms.extend(atoms_secondary)
    atoms = sort(atoms)
    
    return atoms
    

##################
### SUBROUTINE ###
##################


def get_ase_atoms_results_dict(in_atoms):

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

    results["energy_eV"] = in_atoms.get_potential_energy()

    results["stress_voigt_GPa"] = in_atoms.get_stress() * \
        mod_units.eV_per_cubic_Angstrom_to_Pa * (1.0E-9)

    results["volume_A3"] = in_atoms.calc.atoms.get_volume()

    results["cellpar_A"] = \
        geometry.cell.cell_to_cellpar(in_atoms.calc.atoms.get_cell())

    results["cellvecs_A"] = in_atoms.get_cell()[:]

    if 'magmoms' in in_atoms._calc.results.keys():
        results["magmoms"] = in_atoms.get_magnetic_moments()
    else:
        results["magmoms"] = [None]

    results["forces"] = in_atoms.get_forces()

    results["n_atoms"] = len(in_atoms.get_chemical_symbols())

    return results

##################
### SUBROUTINE ###
##################
def print_dft_results_from_ase_atoms(out_atoms, filename="dft_results.json"):

    dft_results = get_ase_atoms_results_dict(out_atoms)

    with open(filename, "w") as fh:
        json.dump(dft_results, fh, indent=4,
                  cls=mod_utility.json_numpy_encoder)

##################
### SUBROUTINE ###
##################


def strain_ase_atoms(in_atoms, strain, strain_type="finite", is_voigt=False):

    mod_utility.error_check_argument_required(
        arg_val=strain_type, arg_name="strain_type", module=module_name,
        subroutine="strain_ase_atoms", valid_args=["finite", "infinitesimal"])

    if strain_type == "infinitesimal":
        return strain_ase_atoms_infinitesimal(in_atoms, strain, is_voigt)

    if strain_type == "finite":
        return strain_ase_atoms_finite(in_atoms, strain, is_voigt)

##################
### SUBROUTINE ###
##################


def strain_ase_atoms_infinitesimal(in_atoms, strain, is_voigt=False):

    if is_voigt:
        strain_matrix = mod_tensor.voigt_6_to_full_3x3_strain(strain)
    else:
        strain_matrix = deepcopy(strain)

    # Strain the cell
    out_cell = np.copy(in_atoms.get_cell())
    for i in range(0, len(out_cell)):
        out_cell[i] = mod_tensor.apply_infinitesimal_strain_to_vector(
            strain_matrix, out_cell[i])

    # Setup output atoms
    out_atoms = deepcopy(in_atoms)
    out_atoms.set_cell(out_cell)
    out_atoms.set_scaled_positions(in_atoms.get_scaled_positions())

    return out_atoms

##################
### SUBROUTINE ###
##################


def strain_ase_atoms_finite(in_atoms, strain, is_voigt=False):

    if is_voigt:
        strain_matrix = mod_tensor.voigt_6_to_full_3x3_strain(strain)
    else:
        strain_matrix = deepcopy(strain)

    # Get the deformation gradient matrix
    F = mod_tensor.convert_strain_to_deformation(strain_matrix)

    # Apply deformation gradient
    return defgrad_ase_atoms(in_atoms, F)

##################
### SUBROUTINE ###
##################


def defgrad_ase_atoms(in_atoms, F):

    # Strain the cell
    out_cell = np.copy(in_atoms.get_cell())
    for i in range(0, len(out_cell)):
        out_cell[i] = mod_tensor.apply_defgrad_to_vector(F, out_cell[i])

    # Setup output atoms
    out_atoms = deepcopy(in_atoms)
    out_atoms.set_cell(out_cell)
    out_atoms.set_scaled_positions(in_atoms.get_scaled_positions())
    return out_atoms

##################
### SUBROUTINE ###
##################


def rescale_ase_atoms_volume(
        in_atoms, volumetric_strain_percent=0.0, target_volume=None):
    '''
    Scales a ase atoms object cell volume
    '''

    # Precedence target_volume > volume_scale_factor
    if target_volume is not None:
        V0 = in_atoms.get_volume()
        volumetric_strain_percent = 100.0 * (target_volume - V0) / V0

    strain_mag = (volumetric_strain_percent / 100.0 + 1)**(1 / 3) - 1.0
    strain_voigt_pattern = np.array([1, 1, 1, 0, 0, 0])
    strain_voigt = strain_mag * strain_voigt_pattern
    strain_matrix = mod_tensor.voigt_6_to_full_3x3_strain(strain_voigt)

    return strain_ase_atoms(in_atoms, strain_matrix)

##################
### SUBROUTINE ###
##################


def get_ase_atoms_at_temperature(in_atoms, qha_flnm, target_temperature):

    # Read qha file
    with open(qha_flnm) as fhandle:
        qha_results = json.load(fhandle)

    f_interp = interp1d(
        qha_results["temperatures_K_list"],
        qha_results["minimum_energy_volumes_A3_list"],
        kind='cubic')

    target_volume = f_interp(target_temperature)
    out_atoms = rescale_ase_atoms_volume(
        in_atoms, target_volume=target_volume)

    return out_atoms

##################
### SUBROUTINE ###
##################

'''
def get_ase_atoms_from_eos(in_atoms, eos_flnm, target_pressure=None, target_temperature=None, interpolate_kind='cubic'):

    # Sanity check
    if (target_temperature is None) and (target_pressure is None):

        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'get_ase_atoms_from_eos'\n")
        sys.stderr.write("       Input arguments either target_temperature or target_presuure or both should be specified\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Read the EOS file and create the EOS object
    EOS_obj = mod_eos_computation.Class_EOS(EOS_filename=eos_flnm, interpolate_kind=interpolate_kind)

    # Get cell parameters
    out_cell = EOS_obj.get_cellvecs(p_GPa=target_pressure)

    # Set cell
    out_atoms = deepcopy(in_atoms)
    out_atoms.set_cell(out_cell)
    out_atoms.set_scaled_positions(in_atoms.get_scaled_positions())

    return out_atoms
'''

##################
### SUBROUTINE ###
##################


def read_qe_input_kpoints(card_lines):
    '''
    Parse kpoints from K_POINTS card.
    '''

    # Remove blanks or comment lines
    trimmed_lines = (line for line in card_lines
                     if line.strip() and not line[0] == '#')

    kpts = None
    for line in trimmed_lines:
        if line.strip().startswith('K_POINTS'):
            if 'automatic' in line.strip():
                split_line = next(trimmed_lines).split()
                kpts = (int(split_line[0]), int(
                    split_line[1]), int(split_line[2]))
                if len(split_line) == 6:
                    koffset = (int(split_line[3]), int(
                        split_line[4]), int(split_line[5]))
                else:
                    koffset = (0, 0, 0)
                return (kpts, koffset)
            else:
                sys.stderr.write("Error: In module mod_ase.py'\n")
                sys.stderr.write(
                    "       In subroutine 'read_espresso_in_kpoints'\n")
                sys.stderr.write("       Can only read 'automatic' K_POINTS\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)

    if kpts is None:
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'read_espresso_in_kpoints'\n")
        sys.stderr.write("       Could not find K_POINTS card\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def read_qe_input_pseudopotentials(card_lines, n_species):
    '''
    Parse kpoints from ATOMIC_SPECIES card.
    '''

    # Remove blanks or comment lines
    trimmed_lines = (line for line in card_lines
                     if line.strip() and not line[0] == '#')

    pseudopotentials = {}
    for line in trimmed_lines:
        if line.strip().startswith('ATOMIC_SPECIES'):
            for sp in range(0, n_species):
                split_line = next(trimmed_lines).split()
                if len(split_line) == 3:
                    pseudopotentials.update({split_line[0]: split_line[2]})
                else:
                    sys.stderr.write("Error: In module mod_ase.py'\n")
                    sys.stderr.write(
                        "       In subroutine 'read_espresso_in_pseudopotentials'\n")
                    sys.stderr.write("       Something wrong\n")
                    sys.stderr.write(
                        "       Other than 3 entries per line encountered\n")
                    sys.stderr.write("       Number of entries: %d\n"
                        %(len(split_line)))
                    print(split_line)
                    sys.stderr.write("       Terminating!!!\n")
                    exit(1)

    if pseudopotentials is None:
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write(
            "       In subroutine 'read_espresso_in_pseudopotentials'\n")
        sys.stderr.write("       Could not find ATOMIC_SPECIES card\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    return pseudopotentials

##################
### SUBROUTINE ###
##################


def read_qe_input(pwi_filename="espresso.pwi"):
    '''
    Reads QE input file and creates the ASE atoms and input calculator
    structures.
    '''

    from ase.calculators.espresso import Espresso
    from ase.io import espresso

    mod_utility.error_check_file_exists(filename=pwi_filename,
                                        module="mod_ase.py",
                                        subroutine="read_qe_input")

    # Read atoms
    atoms_pwi = espresso.read_espresso_in(pwi_filename)

    # Parse namelist section and extract remaining lines
    pwi_filename_demag = os.path.splitext(pwi_filename)[0] + "_demag.pwi"
    copyfile(pwi_filename, pwi_filename_demag)
    # Remove the magnetization line
    mod_utility.replace(file_path=pwi_filename_demag,
                        pattern="starting_magnetization",
                        subst=None,
                        replace_entire_line=True)

    fhandle = open(pwi_filename_demag, 'r')
    input_data, card_lines = espresso.read_fortran_namelist(fhandle)

    # Parse cardlines for kpoints and pseudopotentials
    [kpts, koffset] = read_qe_input_kpoints(card_lines=card_lines)
    n_species = len(set(atoms_pwi.numbers))
    pseudopotentials = read_qe_input_pseudopotentials(
        card_lines=card_lines, n_species=n_species)

    # Create a calculator object
    calc_pwi = Espresso(pseudopotentials=pseudopotentials,
                        input_data=input_data,
                        #input_data=f90nml.read('espresso.pwi'),
                        kpts=kpts,
                        koffset=koffset)

    return (atoms_pwi, calc_pwi)

##################
### SUBROUTINE ###
##################


def read_qe_output_actual(pwo_filename="espresso.pwo"):
    '''
    Reads a QE .pwo output file for results, atoms and calculator
    '''

    from ase.calculators.espresso import Espresso
    from ase import Atom, Atoms

    mod_utility.error_check_file_exists(filename=pwo_filename,
                                        module="mod_ase.py",
                                        subroutine="read_qe_output_calculator")

    # Read the previous output file to get results
    label = os.path.splitext(pwo_filename)[0]
    out_calc = Espresso(label=label)
    out_calc.read_results()

    # Return atoms and calculator object
    return out_calc

##################
### SUBROUTINE ###
##################

'''
def read_qe_output(pwo_filename="espresso.pwo"):

    # Reads a QE .pwo output file for results, atoms and calculator

    from matkit.core import mod_io_espresso_local
    from ase import Atom, Atoms

    mod_utility.error_check_file_exists(filename=pwo_filename,
                                        module="mod_ase.py",
                                        subroutine="read_qe_output_calculator")

    # Read the output file to get results
    gen = mod_io_espresso_local.read_espresso_out(pwo_filename)
    out_calc = next(gen)

    # Return atoms and calculator object
    return out_calc
'''
##################
### SUBROUTINE ###
##################


def read_qe_output_ase_atoms(pwo_filename="espresso.pwo"):
    '''
    Read QE .pwo output file, gets the atoms and sets the magnetic moment to
    the final magnetic moment value.
    '''

    out_calc = read_qe_output(pwo_filename)
    #out_calc = read_qe_output_actual(pwo_filename)

    '''
    # Get new atoms and set magnetic moments
    out_atoms = deepcopy(out_calc.calc.atoms)
    out_atoms.set_initial_magnetic_moments(out_calc.get_magnetic_moments())

    # Return atoms and calculator object
    return out_atoms
    '''
    return qe_output_object_to_ase_atoms(out_calc)

##################
### SUBROUTINE ###
##################


def qe_output_object_to_ase_atoms(out_calc):

    # Get new atoms and set magnetic moments
    out_atoms = deepcopy(out_calc.calc.atoms)

    if "magmoms" in out_calc.calc.results:
        out_atoms.set_initial_magnetic_moments(out_calc.get_magnetic_moments())

    # Return atoms and calculator object
    return out_atoms

'''----------------------------------------------------------------------------
                                 END OF MODULE
----------------------------------------------------------------------------'''
