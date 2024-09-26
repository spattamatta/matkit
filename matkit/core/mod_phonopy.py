'''----------------------------------------------------------------------------
                                mod_phonopy.py

 Description: General routines related to Phonopy.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import re
import sys
import math
import subprocess
import numpy as np
from shutil import copyfile

# Externally installed modules
from ase import Atoms
import phonopy

# Local imports
from matkit.core import mod_units, mod_utility

'''----------------------------------------------------------------------------
                                   MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_phonopy"

class Thermal_Mesh_Info:

    kind = "phonopy"
    dim = None
    mp = None
    force_constants=None
    force_sets=None
    tprop = None
    tmin = None
    tmax = None
    tstep = None

    # parameterized constructor
    def __init__( self,
                  dim = [1, 1, 1],
                  mp = [16, 16, 16],
                  force_constants = None,
                  force_sets=None,
                  tprop = True,
                  tmin = 0.0,
                  tmax = 1000.0,
                  tstep = 10.0,
                ):
        self.dim = dim
        self.mp = mp
        self.force_constants = force_constants
        self.force_sets = force_sets
        self.tprop = tprop
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep

    # Print object
    def print(self, file_name="stdout"):

        if file_name == "stdout":
            fhandle = sys.stdout
        else:
            fhandle = open(file_name, 'w')

        fhandle.write("DIM = " + " ".join(map(str, self.dim)) + "\n")
        fhandle.write("MP = " + " ".join(map(str, self.mp)) + "\n")
        if self.force_constants is not None:
            fhandle.write("FORCE_CONSTANTS = %s\n" %(self.force_constants))
        if self.force_sets is not None:
            fhandle.write("FORCE_SETS = %s\n" %(self.force_sets))
        fhandle.write("TPROP = .%s.\n" %(str(self.tprop).upper()))
        fhandle.write("TMIN = %f\n" %(self.tmin))
        fhandle.write("TMAX = %f\n" %(self.tmax))
        fhandle.write("TSTEP = %f\n" %(self.tstep))

'''----------------------------------------------------------------------------
                                  SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################


def get_minimal_supercell(in_atoms):

    '''
    Computes the minimal supercell needed for phonon convergence
    '''

    cell_params = in_atoms.cell.cellpar()
    min_supercell_length = 8.0 # Minimum 8 Angstroms is needed for phonon convergence
    supercell = np.zeros(3)
    for i in range(0,3):
        supercell[i] = math.ceil(min_supercell_length / cell_params[i])
        
    return supercell

##################
### SUBROUTINE ###
##################
def compute_thermal_properties(work_dir, super_cell, phonopy_thermal_mesh_obj,
                               results_dir="phonopy_results" ):

    old_dir = os.getcwd()
    os.chdir(work_dir)

    phonopy_thermal_mesh_obj.print("mesh.conf")

    super_cell_str = "'" + " ".join(map(str, super_cell)) + "'"

    # Compute thermal properties
    if super_cell is None:
        process = subprocess.call("phonopy -t -p mesh.conf -s", shell=True)
    else:
        process = subprocess.call(
            "phonopy --dim=" +
            super_cell_str +
            " -c POSCAR-unitcell -t -p -v mesh.conf -s",
            shell=True)

    # Compute free energies
    process = subprocess.call("phonopy-vasp-efe --tmax=" +
                              str(phonopy_thermal_mesh_obj.tmax) +
                              " --tstep=" +
                              str(phonopy_thermal_mesh_obj.tstep) +
                              " vasprun.xml", shell=True)

    # Create phonopy_results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Move results to results directory
    result_file_list = [
        "mesh.conf",
        "mesh.yaml",
        "phonopy.yaml",
        "thermal_properties.pdf",
        "thermal_properties.yaml",
        "e-v.dat",
        "fe-v.dat"]
    for result_file in result_file_list:
        os.rename(result_file, results_dir + "/" + result_file)

    [volumes, temperatures, electronic_free_energies, phonon_free_energies] = read_thermal_properties(results_dir, super_cell)

    os.chdir(old_dir)
    return (volumes, temperatures, electronic_free_energies, phonon_free_energies)

##################
### SUBROUTINE ###
##################
def read_thermal_properties(work_dir, super_cell=[1,1,1]):

    from phonopy import file_IO

    super_cell_fac = super_cell[0] * super_cell[1] * super_cell[2]

    # Read e-v file
    [volumes, electronic_energies] = file_IO.read_v_e(work_dir+"/e-v.dat")
    volumes = volumes/super_cell_fac # Just a single element
    electronic_energies = electronic_energies/super_cell_fac # Just a single element

    # Read fe-v.dat
    [temperatures, electronic_free_energies] = file_IO.read_efe(
        work_dir + "/fe-v.dat")
    electronic_free_energies = np.array(electronic_free_energies)[
        :, 0] / super_cell_fac

    # Read thermal_properties.yaml
    [temperatures, cv, entropy, fe_phonon, num_modes, num_integrated_modes] = file_IO.read_thermal_properties_yaml([work_dir+"/thermal_properties.yaml"])
    fe_phonon = np.array(fe_phonon)[:,0]/super_cell_fac

    # Convert to eV
    phonon_free_energies = fe_phonon / mod_units.eV_to_kJ_per_mol

    return (volumes[0], temperatures, electronic_free_energies, phonon_free_energies)

##################
### SUBROUTINE ###
##################


def read_thermal_properties_yaml(src_dir, super_cell=[1,1,1]):

    super_cell_fac = super_cell[0] * super_cell[1] * super_cell[2]

    # Read thermal_properties.yaml
    [temperatures, cv, entropy, fe_phonon, num_modes, num_integrated_modes] = phonopy.file_IO.read_thermal_properties_yaml([src_dir+"/thermal_properties.yaml"])
    fe_phonon = np.array(fe_phonon)[:,0]/super_cell_fac

    # Convert to eV
    phonon_free_energies = fe_phonon / mod_units.eV_to_kJ_per_mol

    return (temperatures, phonon_free_energies)

##################
### SUBROUTINE ###
##################


def read_qe_phonopy_supercell_file(filename):

    '''
    When phonopy generates supercells with dispacements given a qe input file,
    the atom positions can cell details are writen to supercell files. This
    subroutine reads those supercell files to construct ase atoms.
    '''

    with open(filename, "r") as fh:
        # Read first line and get the number of atoms
        line = fh.readline().rstrip()
        line_arr = re.split('[, ""]+', line)
        nat = int(line_arr[6])
        ntyp = int(line_arr[9])

        # Read CELL_PARAMETERS
        line = fh.readline()
        length_units = line.split()[1]
        if length_units == "bohr":
           length_conv = mod_units.bohr_to_angstrom
        elif length_units == "angstrom":
            length_conv = 1.0
        else:
            sys.stderr.write("Error: In module mod_phonopy.py'\n")
            sys.stderr.write("       In subroutine 'read_qe_phonopy_supercell_file'\n")
            sys.stderr.write("       Unknown length unit '%s'\n" % (length_units))
            sys.stderr.write("       Only 'bohr' or 'angstrom' units of length for CELL_PARAMETERS are implemented\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        cell = np.zeros((3, 3))
        for i in range(0, 3):
            cell[i][:] = length_conv*np.array([float(x) for x in fh.readline().split()])

        # Skip ATOMIC_SPECIES
        line = fh.readline() # Skip tag
        for i in range(0, ntyp):
            line = fh.readline()

        # Read ATOMIC_POSITIONS
        atomic_positions = np.zeros((nat, 3))
        chemical_symbols = []
        line = fh.readline() # Skip tag
        for i in range(0, nat):
            line_arr = fh.readline().rstrip().split()
            chemical_symbols.append(line_arr.pop(0))
            atomic_positions[i][:] = np.array([float(x) for x in line_arr])

    out_atoms = Atoms(chemical_symbols, cell=cell,
                      scaled_positions=atomic_positions)

    return out_atoms

##################
### SUBROUTINE ###
##################


def validate_super_cell(super_cell):

    super_cell_str = "'" + " ".join(map(str, super_cell)) + "'"
    if len(super_cell) != 3:
        sys.stderr.write("Error: In module mod_phonopy.py'\n")
        sys.stderr.write("       In subroutine 'validate_super_cell'\n")
        sys.stderr.write("       'super_cell' variable should be a numpy array with 3 "\
            "integer elements\n")
        sys.stderr.write("       Passed super_cell: %s\n" %(super_cell_str))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def setup_magmoms_supercell_from_unitcell(supercell_atoms, magmom_conf, supercell):

    # Validate supercell
    validate_super_cell(supercell)

    # Number of repetitions
    n_rep = supercell[0] * supercell[1] * supercell[2]

    # Read magmom_conf file
    magmom_unitcell = mod_utility.read_array_data(flnm=magmom_conf, n_headers=0, separator=None, data_type="float")[0]
    magmom_supercell = np.tile(magmom_unitcell, n_rep)

    supercell_atoms.set_initial_magnetic_moments(magmom_supercell)

##################
### SUBROUTINE ###
##################

def compute_thermal_properties_qe(
        src_dir="./", dest_dir="./", force_constant_method=None,
        thermal_info=None):
    '''
    Compute thermal properties at a configuration
    '''

    mod_utility.error_check_argument_required(
        force_constant_method, "force_constant_method", module=module_name,
        subroutine="compute_thermal_properties_qe")

    if force_constant_method == "frph":
        compute_thermal_properties_qe_frph(src_dir=sec_dir, dest_dir=dest_dir,
                                           thermal_info=thermal_info)

    elif force_constant_method == "dfpt":
        sys.stderr.write("Error: In module mod_phonopy.py'\n")
        sys.stderr.write("       In subroutine 'compute_thermal_properties_qe'\n")
        sys.stderr.write("       Option force_constant_method: 'dfpt' is not yet implemented\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    else:
        sys.stderr.write("Error: In module mod_phonopy.py'\n")
        sys.stderr.write("       In subroutine 'compute_thermal_properties_qe'\n")
        sys.stderr.write("       Unknown force_constant_method: %s\n"
                         % (force_constant_method))
        sys.stderr.write("       Allowed options: 'dfpt' or 'frph'\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################

def compute_thermal_properties_qe_frph(src_dir="./", dest_dir="./",
                                       thermal_info=None, super_cell=[1,1,1]):

    '''
    Computes thermal properties at a given configuration
    '''

    # List of files needed for computation
    # FORCE_SETS, espresso.in (unitcell not supercell), mesh.conf

    # Create destination directory if it does not already exits
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Write the mesh.conf file in the dest_dir
    if thermal_info.kind == "phonopy":
        thermal_info.print(file_name=dest_dir+"/mesh.conf")

    else:
        sys.stderr.write("Error: In module mod_phonopy.py'\n")
        sys.stderr.write("       In subroutine 'compute_thermal_properties_qe_frph'\n")
        sys.stderr.write("       Input argument 'thermal_info' needs to be of the kind 'phonopy'\n")

    # If dest_dir is different from the src_dir, copy FORCE_SETS
    mod_utility.error_check_file_exists(
        filename=src_dir+"/FORCE_SETS", module=module_name,
        subroutine="compute_thermal_properties_qe_frph")

    copyfile(src_dir+"/FORCE_SETS", dest_dir+"/FORCE_SETS")
    copyfile(src_dir+"/espresso.pwi", dest_dir+"/espresso.pwi")

    # Move to the dest directory
    old_dir = os.getcwd()
    os.chdir(dest_dir)

    # Compute thermal properties
    process = subprocess.call("phonopy --qe -c espresso.pwi -t mesh.conf", shell=True)

    [temperature_list, cv_list, entropy_list, fe_phonon_list, num_modes, num_integrated_modes] = \
       phonopy.file_IO.read_thermal_properties_yaml(["thermal_properties.yaml"])
    fe_phonon_list = fe_phonon_list / mod_units.eV_to_kJ_per_mol
    os.chdir(old_dir)

    # FUTURE: Compute electronic free energies as a function of temperature
    fe_electron_list = [0.0]*len(temperature_list)

    return (temperature_list, cv_list[:,0], entropy_list[:,0], fe_phonon_list[:,0], fe_electron_list)

'''----------------------------------------------------------------------------
                               END OF MODULE
----------------------------------------------------------------------------'''
