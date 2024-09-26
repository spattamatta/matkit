'''----------------------------------------------------------------------------
                               mod_calculator.py

 Description: This is an interface module for various calculators.
              Currently only VASP is available.
              Quantum ESPRESSo, LAMMPS interfaces will be added in the future.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json

# Externally installed modules
# None

# Local imports
from matkit.calculators import mod_vasp
from matkit.core import mod_ase, mod_utility

'''----------------------------------------------------------------------------
                                 MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_calculator.py"

template_sim_info = {
    'calculator' : None,            # Type of calculator: 'vasp', Future: 'qe', 'lammps', 'emto', 'must', ...
    'sim_type' : None,              # Type of simulation: 'as_it_is', 'static', 'relax_i', 'relax_isv', 'relax_is', 'relax_iv', 'dfpt', 'copt', ...
    'n_proc' : 1,                   # Number of processors to be used for the calculation
    'n_relax_iter' : 1,             # Numer of relaxation iterations
    'is_final_static' : False,      # Do a final static (energy calculation) run?
    'is_direct_coordinates' : True, # ASE relates: Are the coordinates direct?
    'sim_specific_input' : None     # Some simulations have additional inputs such as COPT
}

# Dictionary of DFT generic method names for various calculators
calculator_sim_dict = {
    "vasp": {
        "ditto": "ditto",
        "static": "static",
        "relax_i": "isif2",
        "relax_isv": "isif3",
        "relax_is": "isif4",
        "dfpt": "ibrion8",
        "copt" : "ditto"
    },

    "qe": {
        "ditto" : "ditto",
        "static": "scf",
        "relax_i": "relax",
        "relax_isv": "vc-relax",
        "relax_is": None,
        "relax_iv" : "vc-relax-cell_dofree-volume",
        "dfpt": "dfpt"
    },
    
    "must": {
        "ditto" : "ditto",
        "static": "scf"
    }
    
}

'''----------------------------------------------------------------------------
                                  SUBROUTINES
----------------------------------------------------------------------------'''
##################
### SUBROUTINE ###
##################


def run_or_prepare_sim(
        calculator,
        in_atoms=None, in_calc=None, n_proc=1, sim_type=None, n_iter=1,
        is_prepare=False, work_dir=None, final_static=False,
        is_direct_coordinates=True, **kwargs):

    # Check if the sim_type exists for the calculator
    if sim_type in calculator_sim_dict[calculator]:
        calculator_sim_type = calculator_sim_dict[calculator][sim_type]
    else:
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'run_or_prepare_sim'\n")
        sys.stderr.write("       sim_type : %s does not exist for " \
                         "calculator: %s\n" % (sim_type, calculator))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # If a preparation run then returns None, else returns ase atoms
    if calculator == "vasp":
        return mod_vasp.run_or_prepare_sim(
            in_atoms=in_atoms, in_calc=in_calc, n_proc=n_proc,
            sim_type=calculator_sim_type, n_iter=n_iter,
            is_prepare=is_prepare, work_dir=work_dir,
            final_static=final_static,
            is_direct_coordinates=is_direct_coordinates, **kwargs)

    if calculator == "qe":
        return mod_qe.run_or_prepare_sim(
            in_atoms=in_atoms, in_calc=in_calc, n_proc=n_proc,
            sim_type=calculator_sim_type, n_iter=n_iter,
            is_prepare=is_prepare, work_dir=work_dir,
            final_static=final_static,
            is_direct_coordinates=False)

    if calculator == "must":
        return mod_must.run_or_prepare_sim(
            in_atoms=in_atoms, in_calc=in_calc, n_proc=n_proc,
            sim_type=calculator_sim_type, n_iter=n_iter,
            is_prepare=is_prepare, work_dir=work_dir,
            final_static=final_static,
            is_direct_coordinates=False)
#<<---- For crystal / direct 
# coordinates need to fic mod_io_espresso_local.py, it for reason always prints for ATOMIC_POSTIONS in angstrons even if crystal corrdinates are set. Need to fix variable "atomic_positions_str"
          
##################
### SUBROUTINE ###
##################


def run_sim_return_ase_atoms(
        calculator,
        n_proc=1, n_iter=1, final_static=False,
        hostfile=None, wdir=None, parallel_settings=None):

    if calculator == "vasp":
        return mod_vasp.run_vasp_return_ase_atoms(
            n_proc=n_proc, n_iter=n_iter, final_static=final_static,
            hostfile=hostfile, wdir=wdir)

    if calculator == "qe":
        return mod_qe.run_qe_return_ase_atoms(
            n_proc=n_proc, n_iter=n_iter, final_static=final_static,
            hostfile=hostfile, parallel_settings=parallel_settings)
            
    if calculator == "must":
        return mod_must.run_must_return_ase_atoms(
            n_proc=n_proc, n_iter=n_iter, final_static=final_static,
            hostfile=hostfile, wdir=wdir)
            
##################
### SUBROUTINE ###
##################


def read_inputs_get_ase(calculator, work_dir):

    if calculator == "vasp" or calculator == 'vasp_ccr':
        return mod_vasp.read_inputs_get_ase(work_dir)
 
##################
### SUBROUTINE ###
##################


def run_sim_return_pymatgen_results(
        calculaotr,
        n_proc=1, n_iter=1, final_static=False,
        hostfile=None, wdir=None, parallel_settings=None):

    if calculator == "vasp":
        return mod_vasp_pymatgen.run_vasp_return_results(
            n_proc=n_proc, n_iter=n_iter, final_static=final_static,
            hostfile=hostfile, wdir=wdir)
          

##################
### SUBROUTINE ###
##################


def tree_clean(calculator, work_dir="./", file_list=None, clean_level="basic"):

    # Warn before cleaning
    sys.stderr.write("Warning: You have choosen to run 'tree_clean'\n")
    sys.stderr.write("         On the directory '%s'\n" %(work_dir))
    sys.stderr.write("         Press YES to continue with cleaning\n")
    sys.stderr.write("         >>> ")
    if input() != "YES":
        sys.stderr.write("Terminating!!!\n")
        exit(1)
    sys.stderr.write("         Press YES one more time to continue with cleaning\n")
    sys.stderr.write("         >>> ")
    if input() != "YES":
        sys.stderr.write("Terminating!!!\n")
        exit(1)

    if calculator == "vasp":
        return mod_vasp.tree_clean(work_dir, file_list, clean_level)

    if calculator == "qe":
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'tree_clean'\n")
        sys.stderr.write("       Method is not yet implemented for QE\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

##################
### SUBROUTINE ###
##################


def run_dft_job(
        calculator, n_proc=1, n_iter=1, is_final_static=False,
        hostfile=None, wdir=None, parset_filename=None):

    # Check if there is a #WAITING# tag file
    if os.path.isfile("#WAITING#"):
        os.rename("#WAITING#", "#PROCESSING#")
    else:
        return False

    # Run job i.e constant volume ionic relaxation followed by a static
    # calculation
    out_atoms_arr = run_sim_return_ase_atoms(
        calculator=calculator,
        n_proc=n_proc, n_iter=n_iter, final_static=is_final_static,
        hostfile=hostfile, wdir=wdir, parallel_settings=parset_filename)

    if calculator == 'must':
        dft_results = {
        "energy_eV" : out_atoms_arr
        }

        with open('must_results.json', 'w') as fh:
          json.dump(dft_results, fh, indent=4,
                  cls=mod_utility.json_numpy_encoder)
                  
    else:
        # Extract results
       for idx, out_atoms in enumerate(out_atoms_arr, start=0):
            mod_ase.print_dft_results_from_ase_atoms(
                out_atoms, filename="dft_results_" + str(idx) + ".json")

    # Change the processing tag to done
    os.rename("#PROCESSING#", "#DONE#")
    
##################
### SUBROUTINE ###
##################


def run_dft_job_pymatgen(n_proc=1, n_iter=1, is_final_static=False,
                hostfile=None, wdir=None, parset_filename=None):

    # Check if there is a #WAITING# tag file
    if os.path.isfile("#WAITING#"):
        os.rename("#WAITING#", "#PROCESSING#")
    else:
        return False

    # Run job i.e constant volume ionic relaxation followed by a static
    # calculation
    [contcar_list, outcar_list] = run_sim_return_pymatgen_results(
        n_proc=n_proc, n_iter=n_iter, final_static=is_final_static,
        hostfile=hostfile, wdir=wdir, parallel_settings=parset_filename)

    # Extract results
    for idx, (contcar, outcar) in enumerate(zip(contcar_list, outcar_list), start=0):
        mod_pymatgen.print_dft_results_from_pymatgen_results(
            contcar, outcar, filename="dft_results_" + str(idx) + ".json")

    # Change the processing tag to done
    os.rename("#PROCESSING#", "#DONE#")    

##################
### SUBROUTINE ###
##################


def run_case_job(input_filename, n_proc=1, hostfile=None, wdir=None):

    # Check if there is a #WAITING# tag file
    if os.path.isfile("#WAITING#"):
        os.rename("#WAITING#", "#PROCESSING#")
    else:
        return False

    # Run job i.e constant volume ionic relaxation followed by a static
    # calculation
    dummy = mod_case.run_case(input_filename=input_filename, 
        n_proc=n_proc, hostfile=hostfile, wdir=wdir)

    # Change the processing tag to done
    os.rename("#PROCESSING#", "#DONE#")

##################
### SUBROUTINE ###
##################


def extract_dft_job(calulator, is_final_static=False, wdir=None):

    '''
    Just extract results from already run DFT jobs. Normally the routine
    'run_dft_job' takes care of it. But if the user wants to reextract the
    ase atoms from DFT results for some reason and resave the results, this
    routine shall be used. Foe example. If a single job out of a taskfarm
    fails, the user can manually run that single jobs and seamlessly reextract
    the results into files "dft_results_0.json" and if a static calculation
    exists then into "dft_results_1.json".
    '''

    out_atoms_arr = None

    if calculator == "vasp":
        out_atoms_arr = mod_vasp.extract_vasp_return_ase_atoms(final_static=is_final_static, wdir=wdir)

    if calculator == "qe":
        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'extract_dft_job'\n")
        sys.stderr.write("       Not yet implemented for quatum expresso\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Extract results
    for idx, out_atoms in enumerate(out_atoms_arr, start=0):
        mod_ase.print_dft_results_from_ase_atoms(
            out_atoms, filename="dft_results_" + str(idx) + ".json")

##################
### SUBROUTINE ###
##################


def pre_setup_force_constants(
        calculator,
        in_atoms=None, in_calc_fc=None, super_cell=None, work_dir="./",
        job_driver_filename=None, force_constant_method=None,
        parset_filename=None, is_sym=True):
    '''
    Presetup force constant matrix calculation, even before the previous DFT
    calculation for converged initial configuration becomes available. This is
    necessary for batch processing because the settings for force constants
    such as in_calc_fc, super_cell are readily available during the SETUP
    phase from the material file.
    '''

    # Check for required arguments
    #mod_utility.error_check_argument_required(
    #    arg_val=job_driver_filename, arg_name="job_driver_filename",
    #    module=module_name, subroutine="pre_setup_force_constants")

    mod_utility.error_check_argument_required(
        arg_val=force_constant_method, arg_name="force_constant_method",
        module=module_name, subroutine="pre_setup_force_constants",
        valid_args=["dfpt", "frph"])

    # Calculator: VASP
    if calculator == "vasp" or calculator == "vasp_ccr":

        # Method: DFPT
        if force_constant_method == "dfpt":
            return mod_vasp.pre_setup_dfpt(
                in_atoms=in_atoms, in_calc_fc=in_calc_fc,
                super_cell=super_cell, work_dir=work_dir,
                job_driver_filename=job_driver_filename, is_sym=is_sym)

        # Method: FRPH
        if force_constant_method == "frph":
            sys.stderr.write("Error: In module '%s'\n" %(module_name))
            sys.stderr.write(
                "       In subroutine 'pre_setup_force_constants'\n")
            sys.stderr.write(
                "       Frozen Phonon method is not yet implemented for VASP\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

    # Calculator: Quantum Espresso
    if calculator == "qe":

        # Method: DFPT
        # NOTE: QE - DFPT does not require a super cell
        if force_constant_method == "dfpt":
            return mod_qe.pre_setup_dfpt(
                in_calc_fc=in_calc_fc, work_dir=work_dir,
                job_driver_filename=job_driver_filename)

        # Method: FRPH
        if force_constant_method == "frph":
            return mod_qe.pre_setup_frozen_phonon(
                in_atoms=in_atoms, in_calc_fc=in_calc_fc,
                super_cell=super_cell, work_dir=work_dir,
                job_driver_filename=job_driver_filename)

##################
### SUBROUTINE ###
##################


def setup_force_constants(
        calculator,
        in_atoms=None, in_calc_fc=None, super_cell=None, src_dir="./",
        dest_dir="./", force_constant_method=None):
    '''
    Sets up force constant calculation files once the starting configuration
    becomes available through a previous DFT calculation.
    '''

    # Calculator: VASP
    if calculator == "vasp":

        # Method: DFPT
        if force_constant_method == "dfpt":
            return mod_vasp.setup_dfpt(
                in_atoms=in_atoms, in_calc_fc=in_calc_fc,
                super_cell=super_cell, src_dir=src_dir, dest_dir=dest_dir)

        # Method: FRPH
        if force_constant_method == "frph":
            sys.stderr.write("Error: In module '%s'\n" %(module_name))
            sys.stderr.write("       In subroutine 'setup_force_constants'\n")
            sys.stderr.write(
                "       Frozen Phonon method is not yet implemented for VASP\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

    # Calculator: Quantum Espresso
    if calculator == "qe":

        # Method: DFPT
        # NOTE: QE - DFPT does not require a super cell
        if force_constant_method == "dfpt":
            return mod_qe.setup_dfpt(
                in_atoms=in_atoms, in_calc_fc=in_calc_fc, src_dir=src_dir,
                dest_dir=dest_dir)

        # Method: FRPH
        if force_constant_method == "frph":
            return mod_qe.setup_frozen_phonon(
                in_atoms=in_atoms, in_calc_fc=in_calc_fc,
                super_cell=super_cell, src_dir=src_dir, dest_dir=dest_dir)

##################
### SUBROUTINE ###
##################


def run_force_constants(calculator, work_dir=None, super_cell=None, n_proc=1):
    '''
    Runs force constant calculations.

    FUTURE: Frozen phonon calculations need to be parallelized using taskfarm.
    '''

    # Calculator: VASP
    if calculator == "vasp":

        # Method: DFPT
        if force_constant_method == "dfpt":
            return mod_vasp.run_dfpt(work_dir=work_dir, super_cell=super_cell,
                                     n_proc=n_proc)

        # Method: FRPH
        if force_constant_method == "frph":
            sys.stderr.write("Error: In module '%s'\n" %(module_name))
            sys.stderr.write("       In subroutine 'run_force_constants'\n")
            sys.stderr.write(
                "       Frozen Phonon method is not yet implemented for VASP\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

    # Calculator: Quantum Espresso
    if calculator == "qe":

        # Method: DFPT
        # NOTE: QE - DFPT does not require a super cell
        if force_constant_method == "dfpt":
            return mod_qe.run_dfpt(work_dir=work_dir, n_proc=n_proc)

        # Method: FRPH
        if force_constant_method == "frph":
            return mod_qe.run_frozen_phonon(work_dir=work_dir, n_proc=n_proc)

##################
### SUBROUTINE ###
##################


def setup_stress_constraint_file(calculator, work_dir, mode_id, flag_voigt, stress_voigt):

    '''
    Setup stress file
    '''

    # Calculator: VASP
    if calculator == "vasp":
           
        mod_vasp.setup_constrained_relaxation_file(
            work_dir=work_dir, mode_id=mode_id, flag_voigt_list=flag_voigt,
            value_voigt_list=stress_voigt)

    # Calculator: Quantum Espresso
    if calculator == "qe":

        sys.stderr.write("Error: In module '%s'\n" %(module_name))
        sys.stderr.write("       In subroutine 'setup_stress_constraint_file'\n")
        sys.stderr.write("       Method is not yet implemented for QE\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

'''----------------------------------------------------------------------------
                                 END OF MODULE
----------------------------------------------------------------------------'''
