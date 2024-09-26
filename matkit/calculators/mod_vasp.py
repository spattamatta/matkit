'''----------------------------------------------------------------------------
                                mod_vasp.py

 Description: This module contains all VASP related routines that setup, run,
              parse results.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json
import math
import warnings
import subprocess
import numpy as np
from copy import deepcopy
from shutil import copyfile

# Externally installed modules
from ase import Atom, Atoms
from ase.calculators.vasp import Vasp
# To write POSCAR (gives option to turn on direct coordinates option)
from ase.io.vasp import write_vasp, read_vasp_out, read_vasp

# Local imports
from matkit import CONFIG
from matkit.core import mod_utility, mod_units, mod_batch

'''----------------------------------------------------------------------------
                                 MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_vasp.py"

console_redirect_filename = "console.report.vasp"

# Clean basis run related files
basic_clean_file_list = ['CHG', 'CHGCAR', 'DOSCAR', 'EIGENVAL', 'IBZKPT',
    'PCDAT', 'WAVECAR', 'XDATCAR', 'PROCAR', 'REPORT', 'LOCPOT', 'AECCAR0',
    'AECCAR1', 'AECCAR2', 'ase-sort.dat']

# Clean all run related files and results as if run has not taken place
dist_clean_file_list = basic_clean_file_list + ['OUTCAR', 'OSIZCAR', 'CONTCAR',
    'vasprun.xml']
    
# Saving results for future
saver_clean_file_list = basic_clean_file_list + ['OUTCAR', 'OSIZCAR',
   'CONTCAR', 'vasprun.xml', 'POTCAR']

bug_list = ["ZBRENT: fatal error in bracketing", \
            "ZBRENT: fatal error: bracketing interval incorrect", \
            "Inconsistent Bravais lattice"]

'''
calc_defaults: These are the set of default values for various calculation
    setups. If the priority value of a calc_defaults field is "High" then it
    overwrites the coreesponding filed in the calculator. If the priority is
    "Default", then it is written to the calculator only if it does not exist
    in the claculator setup. See subroutine: set_defaults_or_overwrite.

In case of VASP each default field is an input variable in the input file. The
dictionary of this variable contains 3 members. The first member is the value
of the variable, the second is the type of the variable internal to ase's
class and the third is the priority as described above.
'''

# ISIF 2: Hold volume and cell shape constant and relax ions
isif2_calc_defaults = {
    "ibrion": [2, "int_params", "Default"],
    "isif": [2, "int_params", "High"],
    "nsw": [100, "int_params", "Default"],
    "prec": ["Accurate", "string_params", "Default"],
    "addgrid": [True, "bool_params", "Default"],
    "ediff": [1.0E-6, "exp_params", "Default"],
    "ediffg": [-0.01, "exp_params", "Default"]
}

# ISIF 3: Relax ions, shape and volume simultaneously
isif3_calc_defaults = {
    "ibrion": [2, "int_params", "High"],
    "isif": [3, "int_params", "High"],
    "nsw": [100, "int_params", "Default"],
    "prec": ["Accurate", "string_params", "Default"],
    "addgrid": [True, "bool_params", "Default"],
    "ediff": [1.0E-6, "exp_params", "Default"],
    "ediffg": [-0.01, "exp_params", "Default"]
}

# ISIF 4: Hold volume and relax both cell shape and ions
isif4_calc_defaults = {
    "ibrion": [2, "int_params", "Default"],
    "isif": [4, "int_params", "High"],
    "nsw": [100, "int_params", "Default"],
    "prec": ["Accurate", "string_params", "Default"],
    "addgrid": [True, "bool_params", "Default"],
    "ediff": [1.0E-6, "exp_params", "Default"],
    "ediffg": [-0.01, "exp_params", "Default"]
}

# static: Calculations of the DOS and very accurate total energy calculations
#         (no relaxation). For metals use the tetrahedron method ISMEAR=-5
static_calc_defaults = {
    "ibrion": [-1, "int_params", "High"],
    "ismear": [-5, "int_params", "High"],
    "nsw": [0, "int_params", "High"]
}

# IBRION 8: Second derivatives, Hessian matrix and phonon frequencies
# (perturbation theory)
ibrion8_calc_defaults = {
    "ibrion": [8, "int_params", "High"],
    "ismear": [0, "int_params", "High"],
    "isif" : [None, "int_params", "High"],
    "nsw": [1, "int_params", "High"],
    "ialgo": [38, "int_params", "High"],
    "prec": ["Accurate", "string_params", "High"],
    "lreal": [False, "bool_params", "High"],
    "addgrid": [True, "bool_params", "High"],
    "lwave": [False, "bool_params", "High"],
    "lcharg": [False, "bool_params", "High"],
    "ediff": [1.0E-8, "exp_params", "High"],
    "ediffg": [-1.0E-7, "exp_params", "High"]
}


# IBRION 7: Same as ibrion8 but without symmetry
ibrion7_calc_defaults = deepcopy(ibrion8_calc_defaults)
ibrion7_calc_defaults["ibrion"] = [7, "int_params", "High"]

'''----------------------------------------------------------------------------
                                   SUBROUTINE
----------------------------------------------------------------------------'''


def read_inputs_get_ase(work_dir):

    # Read the POSCAR and get ase atoms
    atoms_in = read_vasp(work_dir + "/POSCAR")
    return atoms_in

##################
### SUBROUTINE ###
##################


def setup_vasp_calc(system=None, encut=None,  setups=None,
                    kspacing=2.0 * math.pi * 0.02, kpoints=None,
                    sigma=0.1, isif=3, ibrion=2, ediff=1.0E-8, ediffg=-1.0E-4,
                    isym=2, is_vca=False, sorted_mixing_fraction_arr=None):

    '''
    Basic setup for VASP calculations
    '''

    # ASE VASP calculator settings
    calc = Vasp(
        # Pseudo potential parameters
        xc='PBE',
        gga='PE',
        # Start parameters for this run
        prec='Accurate',   # High depends on ENAUG
        istart=0,          # Job   : 0-new  1-cont  2-samecut, Do not read WAVECAR as the system is under incremnetal loading
        icharg=1,          # Charge: 1-file 2-atom 10-const
        addgrid=True,
        lasph=True,        # Aspherical contributions
        ispin=2,
        lorbit=10,
        # Electronic relaxation parametrs
        nelm=100,          # Max SCF steps
        nelmin=5,          # Min SCF steps
        ediff=ediff,       # Stopping criterion for ELM
        lreal=False,
        voskown=1,
        algo='Normal',
        nelmdl=-5,
        # Ionic relaxation parameters
        nsw=60,            # Number of steps for ionic motion
        isif=isif,
        ediffg=ediffg,
        ibrion=ibrion,
        potim=0.1,         # Time step for ionic motion
        isym=isym,
        # DOS related                
        ismear=1,          # 1st order Methfessel-Paxton, -5 tetrahedron 
        sigma=sigma,       # Broadening in eV
        # Write flags
        lwave=False,       # Write WAVECAR
        lcharg=True,       # Write CHGCAR
        lelf=False,        # Write electronic localization function (ELF)
        symprec=1.0E-8     
    )

    # Comment: Name of the calculation (Start parameter)
    if system is not None:
        calc.set(system=system)
        
    # If encut is not provided uses default encut in the pseudo potenial (Part of electronic relaxation parameter)
    if encut is not None:
        calc.set(encut=encut)   
        
    # If semi core potentials are needed (Part of electronic relaxation parameter)
    if setups is not None:
        calc.set(setups=setups)             

    # Kpoints related: kspacing is the preferred method
    calc.set(gamma=True)
    if kpoints is None:
        calc.set(kspacing=kspacing)
    else:
        calc.set(kpts=kpoints)
        
    # Mixing fraction
    if sorted_mixing_fraction_arr is not None:
        calc.set(vca=sorted_mixing_fraction_arr)

    return calc

##################
### SUBROUTINE ###
##################


def get_sim_type_from_isif(isif):

    if isif == 2:
        return "relax_i"
    elif isif == 3:
        return "relax_isv"
    elif isif == 4:
        return "relax_is"

##################
### SUBROUTINE ###
##################


def run_vasp(n_proc=1, hostfile=None, wdir=None, n_trials=2):

    # Check basis required files
    mod_utility.error_check_file_exists(filename='INCAR', module=module_name, subroutine='run_vasp')
    mod_utility.error_check_file_exists(filename='POSCAR', module=module_name, subroutine='run_vasp')
    mod_utility.error_check_file_exists(filename='POTCAR', module=module_name, subroutine='run_vasp')

    [PATH_VASP_STD, PATH_VASP_GAM, PATH_VASP_NCL, VASP_MPI_TAGS, VASP_ENV_TAGS] = CONFIG.get_vasp_environ()

    if n_proc == 1:
        vasp_command = PATH_VASP_STD
    else:
        vasp_command = mod_batch.mpi_command(n_proc, hostfile, wdir) + \
                       PATH_VASP_STD

    for i in range(0, n_trials):

        process = subprocess.call(vasp_command + " > " + console_redirect_filename,
                                  shell=True)

        # Check for bug
        bug_ids = check_bugs( )
        if len(bug_ids) == 0:
            break
        else:
            bug_handling(bug_ids)

##################
### SUBROUTINE ###
##################


def check_bugs( ):

    bug_ids = []

    # Bugs that reflect in console file
    with open(console_redirect_filename) as fh:
        # Each line
        for line in fh:
            # Match each line with all bugs
            for bug_idx, bug in enumerate(bug_list, start=0):
                if bug in line:
                    bug_ids.append(bug_idx)

    # Unique bug list
    bug_ids = list(set(bug_ids))

    return bug_ids

##################
### SUBROUTINE ###
##################


def bug_handling(bug_ids):

    for bug_idx in bug_ids:

        bug = bug_list[bug_idx]

        if bug == bug_list[0] or bug == bug_list[1]:
            bug_handling_ZBRENT_BRACKETING( )
        elif bug == bug_list[2]:
            bug_handling_BRAVAIS( )

##################
### SUBROUTINE ###
##################


def set_defaults_or_overwrite(in_calc, defaults_calc, **kwargs):
    '''
    1. Highest precedence is given to defaults with explicit mention of"High"
       in precendence
    2. The next is to kwargs and then followed by the existing entries of
       in_calc
    3. Finally if in_calc and kargs donot set field corresponding to "Default"
       precedence in defaults the these defauls are used

    To achieve this the kwargs are constantly updated
    '''

    # STEP 1: Owerwrite kwargs with the highest precedence tags from defaults
    # STEP 2: For tags missing in both kwargs and in_calc but that are in
    # defaults use defaults
    for key in defaults_calc:
        if defaults_calc[key][2] == "High":
            kwargs[key] = defaults_calc[key][0]
        else:
            if (key not in kwargs) and (
                    getattr(in_calc, defaults_calc[key][1])[key] is None):
                kwargs[key] = defaults_calc[key][0]

    in_calc.set(**kwargs)

##################
### SUBROUTINE ###
##################


def run_or_prepare_sim(
        in_atoms=None, in_calc=None, n_proc=1, sim_type=None, n_iter=1,
        is_prepare=False, work_dir=None, final_static=False,
        is_direct_coordinates=True, **kwargs):

    if work_dir is not None:
        mod_utility.error_check_path_exists(
            pathname=work_dir,
            module="mod_vasp.py",
            subroutine="run_or_prepare_sim")
        old_dir = os.getcwd()
        os.chdir(work_dir)

    # Delete default files
    in_calc.clean()
    run_label = "run_" + str(sim_type)
    calc = deepcopy(in_calc)

    if sim_type == "ditto":
        pass
    elif sim_type == "static":
        set_defaults_or_overwrite(calc, static_calc_defaults, **kwargs)
    elif sim_type == "isif1":
        sys.stderr.write("Error: In module 'mod_vasp.py'\n")
        sys.stderr.write("       In subroutine 'run_or_prepare_sim'\n")
        sys.stderr.write("       sim_type = isif1 not yet implemented\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
    elif sim_type == "isif2":
        set_defaults_or_overwrite(calc, isif2_calc_defaults, **kwargs)
    elif sim_type == "isif3":
        set_defaults_or_overwrite(calc, isif3_calc_defaults, **kwargs)
    elif sim_type == "isif4":
        set_defaults_or_overwrite(calc, isif4_calc_defaults, **kwargs)
    elif sim_type == "isif5":
        sys.stderr.write("Error: In module 'mod_vasp.py'\n")
        sys.stderr.write("       In subroutine 'run_or_prepare_sim'\n")
        sys.stderr.write("       sim_type = isif1 not yet implemented\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
    elif sim_type == "isif6":
        sys.stderr.write("Error: In module 'mod_vasp.py'\n")
        sys.stderr.write("       In subroutine 'run_or_prepare_sim'\n")
        sys.stderr.write("       sim_type = isif1 not yet implemented\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
    elif sim_type == "isif7":
        sys.stderr.write("Error: In module 'mod_vasp.py'\n")
        sys.stderr.write("       In subroutine 'run_or_prepare_sim'\n")
        sys.stderr.write("       sim_type = isif1 not yet implemented\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)
    elif sim_type == "ibrion7":
        set_defaults_or_overwrite(calc, ibrion7_calc_defaults, **kwargs)        
    elif sim_type == "ibrion8":
        set_defaults_or_overwrite(calc, ibrion8_calc_defaults, **kwargs)
    else:
        sys.stderr.write("Error: In module 'mod_vasp.py'\n")
        sys.stderr.write("       In subroutine 'run_or_prepare_sim'\n")
        sys.stderr.write("       Unknown sim_type\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Write inputs
    calc.initialize(in_atoms)
    calc.write_input(in_atoms)
    #write_vasp("POSCAR", in_atoms, label=run_label, sort=True, vasp6=True,
    #           direct=is_direct_coordinates)
    write_vasp("POSCAR", in_atoms, sort=True, vasp6=True,
               direct=is_direct_coordinates)

    # If a final static run is required
    if final_static:
        open("#DO_FINAL_STATIC#", 'a').close()

    # If run and not prepare then run the simulation
    out_atoms = None
    if not is_prepare:
        out_atoms = run_vasp_return_ase_atoms(n_proc=n_proc, n_iter=n_iter)

    # Important: Change to old dir
    if work_dir is not None:
        os.chdir(old_dir)

    # If just a preparation run, retun value is None else returns out_atoms
    return out_atoms

##################
### SUBROUTINE ###
##################


def read_outcar_update_magmoms_incar():
    '''
    Reads the OUTCAR file to get the final magnetic moments and updates the
    INCAR file with the magnetic moments
    '''
    # Read OUTCAR
    out_atoms = read_vasp_out("OUTCAR")
    if 'magmoms' in out_atoms.calc.results:
        magmom_str = " MAGMOM = " + \
            " ".join(map(str, out_atoms.get_magnetic_moments()))

        # Delete existing MAGMOM line if at all it exists. This is done as opposed
        # to direct replacement because some INCAR files may not have MAGMOM
        mod_utility.replace(
            file_path="INCAR",
            pattern="MAGMOM",
            subst=None,
            replace_entire_line=True)

        # Append magmom_str to INCAR
        with open("INCAR", "a") as fhandle:
            fhandle.write(magmom_str + "\n")

##################
### SUBROUTINE ###
##################


def bug_handling_ZBRENT_BRACKETING( ):

    '''
    This subroutine handles the following error printed to the file
    console.report.vasp

    ZBRENT: fatal error in bracketing
        please rerun with smaller EDIFF, or copy CONTCAR
        to POSCAR and continue

    ZBRENT: fatal error: bracketing interval incorrect
        please rerun with smaller EDIFF, or copy CONTCAR
        to POSCAR and continue

    To handle this, firstly:
    -----------------------

    The Conjugate gradient algorithm corresponding to IBRION=2 in INCAR,
    searches for the lowest energy configuration, by iteratively computing the
    force on atom, multiplying it by POTIM to move the ions. If the energy at
    the new configuration is lower repeat the above process until convergence
    criterion is met or else estimate the lowest possible point and repeat.

    In large, flexible materials with many degrees of freedom, the CG 
    optimization algorithm ( IBRION=2 ) oftentimes results in a bracketing
    error once it gets relatively close to the local minimum (search for
    ZBRENT: fatal error in bracketing in the standard output file) This occurs
    because the potential energy surface is very flat, and the CG algorithm
    implemented in VASP is based on the energy differences. One option to fix
    this is to copy the CONTCAR to the POSCAR and tighten EDIFF to 1e-6, but a
    more reliable option is to use a force-based optimizer, IBRION = 1, quasi-
    Newton has a small search range.

    The following actions are taken by this subroutine:

    1. Creates a backup of POSCAR as PRE_ZBRENT.POSCAR
    2. Copies CONTCAR to POSCAR
    3. Creates a backup of INCAR as PRE_ZBRENT.INCAR
    4. Modifications to INCAR
       a. IBRION = 1
       b. EDIFF = 1.0E-6
       c. EDIFFG = -1.0E-5
    '''

    # Create POSCAR backup
    os.rename("POSCAR", "PRE_ZBRENT.POSCAR")

    # Copy CONTCAR to POSCAR
    copyfile("CONTCAR", "POSCAR")

    # Create a backup of INCAR
    copyfile("INCAR", "PRE_ZBRENT.INCAR")

    # Modify INCAR
    mod_utility.replace(file_path="INCAR", pattern="IBRION",
                        subst=" IBRION = 1", replace_entire_line=True)

    '''
    # NOTE: Space of string EDIFF is must as it can also match with EDIFFG
    mod_utility.replace(file_path="INCAR", pattern="EDIFF ",
                        subst=" EDIFF = 1.0E-5", replace_entire_line=True)
    mod_utility.replace(file_path="INCAR", pattern="EDIFFG",
                        subst=" EDIFFG = -1.0E-2", replace_entire_line=True)
    '''
    # Update magnetic moments
    read_outcar_update_magmoms_incar()
    
##################
### SUBROUTINE ###
##################


def bug_handling_BRAVAIS( ):

    '''
    This subroutine handles the following error printed to the file
    console.report.vasp

      Inconsistent Bravais lattice types found for crystalline and  reciprocal
      lattice:  
  
 
      In most cases this is due to inaccuracies in the specification of the
      crytalline lattice vectors.

      Suggested SOLUTIONS: 
        a. Refine the lattice parameters of your structure
        b. try changing SYMPREC.

    '''

    # Create a backup of INCAR
    copyfile("INCAR", "PRE_BRAVAIS.INCAR")

    # Modify INCAR
    mod_utility.replace(file_path="INCAR", pattern="SYMPREC",
                        subst=" SYMPREC = 1.0E-6", replace_entire_line=True)

##################
### SUBROUTINE ###
##################


def modify_incar_for_static():

    # Delete IBRION, ISIF, ISMEAR and NSW
    mod_utility.replace(file_path="INCAR", pattern="IBRION", subst=None,
                        replace_entire_line=True)
    mod_utility.replace(file_path="INCAR", pattern="ISIF", subst=None,
                        replace_entire_line=True)
    mod_utility.replace(file_path="INCAR", pattern="ISMEAR", subst=None,
                        replace_entire_line=True)
    mod_utility.replace(file_path="INCAR", pattern="NSW", subst=None,
                        replace_entire_line=True)

    # Append static defaults, look at module variable static_calc_defaults
    '''
    static_calc_defaults = {
        "ibrion"  : [-1,          "int_params",    "High"   ], \
        "ismear"  : [-5,          "int_params",    "High"   ], \
        "nsw"     : [0,           "int_params",    "Default"]
   '''
    with open("INCAR", "a") as fhandle:
        fhandle.write("\n IBRION = -1\n")
        fhandle.write(" ISMEAR = -5\n")
        fhandle.write(" NSW = 0")

##################
### SUBROUTINE ###
##################


def run_vasp_return_ase_atoms(n_proc=1, n_iter=1, final_static=False,
                              hostfile=None, wdir=None):
    '''
    Runs a VASP job in a directory for one or more iterations. If required also
    does a final static calculation. The final static calculation can be
    triggered either by the presence of a '#DO_FINAL_STATIC#' tag file in the
    current directory or explicitly setting to True, the input argument
    final_static.
    '''

    # Run VASP for the first time
    run_vasp(n_proc=n_proc, hostfile=hostfile, wdir=wdir)

    # Run the next (n_iter-1) times
    if (n_iter - 1) >= 1:
        for i in range(n_iter - 1):
            # Read OUTCAR and get the magnetization, include it in INCAR
            read_outcar_update_magmoms_incar()
            # Copy CONTCAR to POSCAR
            process = subprocess.call("cp CONTCAR POSCAR", shell=True)
            # Run vasp
            run_vasp(n_proc=n_proc, hostfile=hostfile, wdir=wdir)

    # This is the return point if no static calculation is to be performed
    out_atoms_arr = [read_vasp_out("OUTCAR")]

    # Check if a final static run is required
    if os.path.isfile("#DO_FINAL_STATIC#") or final_static:
        # Create a static directory
        if not os.path.isdir("FINAL_STATIC"):
            os.makedirs("FINAL_STATIC")
        # Copy necessary files
        # Copy hostfile if it exists
        if hostfile is not None:
            copyfile(hostfile, "FINAL_STATIC/"+hostfile)
        copyfile("CONTCAR", "FINAL_STATIC/POSCAR")
        copyfile("POTCAR", "FINAL_STATIC/POTCAR")  
        if os.path.isfile("KPOINTS"):
            copyfile("KPOINTS", "FINAL_STATIC/KPOINTS")                     
        # INCAR needs to be modified
        copyfile("INCAR", "INCAR_OLD")            
        # Read OUTCAR and get the magnetization, include it in INCAR
        read_outcar_update_magmoms_incar()
        # Include static defaults into the INCAR and run vasp
        modify_incar_for_static()
        os.rename("INCAR", "FINAL_STATIC/INCAR")
        # Reset INCAR in the root directory to original name
        os.rename("INCAR_OLD", "INCAR")
        os.chdir("FINAL_STATIC")                   
        run_vasp(n_proc=n_proc, hostfile=hostfile, wdir=wdir+'/FINAL_STATIC')
        out_atoms_arr.append(read_vasp_out("OUTCAR"))
        os.chdir("../")

    # Return the final calculator that has results
    return out_atoms_arr

##################
### SUBROUTINE ###
##################


def extract_vasp_return_ase_atoms(final_static=False, wdir=None):

    '''
    Read already done computation results. This functionality is needed
    sometimes, see explanation in subroutine mod_calculator.extract_dft_job.
    '''

    # This is the return point if no static calculation is to be performed
    out_atoms_arr = [read_vasp_out("OUTCAR")]

    # Check if a final static run is required
    if os.path.isfile("#DO_FINAL_STATIC#") or final_static:
        os.chdir("FINAL_STATIC")                   
        out_atoms_arr.append(read_vasp_out("OUTCAR"))
        os.chdir("../")

    # Return the final calculator that has results
    return out_atoms_arr

##################
### SUBROUTINE ###
##################


def pre_setup_dfpt(in_atoms, in_calc_fc, super_cell, work_dir,
                   job_driver_filename=None, is_sym=True):
    '''
    Pre-setup for a DFPT calculation, even before the previous DFT calculation
    for converged configuration becomes available.
    '''

    old_dir = os.getcwd()
    os.chdir(work_dir)

    # Create DFPT files
    if os.path.isfile("KPOINTS"):
        os.rename("KPOINTS", "KPOINTS_RELAX")

    os.rename("INCAR", "INCAR_RELAX")
    os.rename("POSCAR", "POSCAR_RELAX")
    if is_sym:
        run_or_prepare_sim(in_atoms=in_atoms, in_calc=in_calc_fc, 
            sim_type="ibrion8", is_prepare=True)  # Poscar is dummy
    else:
        run_or_prepare_sim(in_atoms=in_atoms, in_calc=in_calc_fc, 
            sim_type="ibrion7", is_prepare=True)  # Poscar is dummy

    if os.path.isfile("KPOINTS"):
        os.rename("KPOINTS", "KPOINTS_DFPT")

    os.rename("INCAR", "INCAR_DFPT")
    os.remove("POSCAR")  # Delete POSCAR as we will later use CONTCAR from relaxation run

    if os.path.isfile("KPOINTS_RELAX"):
        os.rename("KPOINTS_RELAX", "KPOINTS")

    os.rename("INCAR_RELAX", "INCAR")
    os.rename("POSCAR_RELAX", "POSCAR")

    # Setup the supercell file
    if super_cell is not None:
        with open("super_cell.conf", "w") as fh:
            fh.write("%d %d %d" % (super_cell[0], super_cell[1], super_cell[2]))

    # Append processing instruction to the job driver in work_dir
    if job_driver_filename is not None:

        with open(job_driver_filename, 'a') as fh:
            fh.write("from matkit.calculators import mod_vasp\n")
            fh.write("mod_vasp.setup_dfpt(src_dir='%s', dest_dir='%s', hostfile=hostfile)\n" % 
                     (work_dir, dfpt_dir))
            fh.write("mod_vasp.run_dfpt(work_dir='%s', n_proc=n_proc, " \
                     "hostfile=hostfile)\n" % (dfpt_dir))

    # Return to old_dir
    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def setup_dfpt(in_atoms=None, in_calc_fc=None, super_cell=None, src_dir="./",
               dest_dir="./", hostfile=None, job_driver_filename='job_driver.py'):
    '''
    Sets up DFPT calculation for a given configuration
    Precedences:
        Creating POSCAR-unitcell:
            in_atoms > CONTCAR (src_dir) > OUTCAR (src_dir) > Error
        Creating INCAR: in_calc_fc > INCAR_DFPT (src_dir) > Error
        Creating KPOINTS: in_calc_fc > KPOINTS_DFPT > KPOINTS (src_dir)
                          > Error
        Creating POTCAR:
            if both in_atoms and dfpt_incar POTCAR autogenerated > 
                POTCAR (src_dir) > Error
        Creating magmom.conf (optional):
            in_atoms > OUTCAR > magmom.conf (src_dir)
        Creating super_cell.conf (optional):
            super_cell > super_cell.conf > [2, 2, 2] (default)
    '''
    # Logical variables to check if KPOINTS and POTCAR are created
    kpoints_created = False
    potcar_created = False

    # Setup directories
    old_dir = os.getcwd()

    # If dest_dir is not given assume "./" as dest_dir else create dest_dir if
    # it does not already exist
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    # Move to src_dir, and always be on it (is it needed ??? in case of
    # in_atoms and in_calc_fc, super_cell provided )
    os.chdir(src_dir)

    # If both in_atoms and in_calc_fc are present
    if (in_atoms is not None) and (in_calc_fc is not None):

        # Prepare both POSCAR-unitcell, INCAR, KPOINTS and POTCAR in dest_dir
        run_or_prepare_sim(
            in_atoms=in_atoms,
            in_calc=in_calc_fc,
            sim_type="ibrion8",
            is_prepare=True,
            work_dir=dest_dir)
        os.rename(dest_dir + "/POSCAR", dest_dir + "/POSCAR-unitcell")
        kpoints_created = True
        potcar_created = True

    else:

        # Setup POSCAR-unitcell
        if in_atoms is not None:
            write_vasp(
                dest_dir +
                "/POSCAR-unitcell",
                in_atoms,
                label="dfpt",
                sort=True,
                vasp5=True,
                direct=True)
        else:
            # Check if src_dir has CONTCAR
            if os.path.isfile("CONTCAR"):
                copyfile("CONTCAR", dest_dir + "/POSCAR-unitcell")
            # Check if src_dir has OUTCAR
            elif os.path.isfile("OUTCAR"):
                out_atoms = read_vasp_out("OUTCAR")
                write_vasp(
                    dest_dir +
                    "/POSCAR-unitcell",
                    in_atoms,
                    label="dfpt",
                    sort=True,
                    vasp5=True,
                    direct=True)
            else:
                sys.stderr.write("Error: In module mod_vasp.py'\n")
                sys.stderr.write(
                    "       In subroutine 'setup_dfpt'\n")
                sys.stderr.write(
                    "       Provide any of in_atoms, src_dir/CONTCAR or \
                    src_dir/OUTCAR to be able to setup POSCAR-unitcell for \
                    dfpt calculations\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)

        # Setup INCAR_DFPT
        if in_calc_fc is not None:
            # This is not properly implemented,redo it
            # Assumes OUTCAR is present
            out_atoms = read_vasp_out("OUTCAR")
            os.chdir(dest_dir)
            in_calc_dpft.initialize(out_atoms)
            in_calc_dpft.write_input(in_atoms)
            os.chdir(src_dir)
            kpoints_created = True
            potcar_created = True

        elif os.path.isfile("INCAR_DFPT"):
            copyfile("INCAR_DFPT", dest_dir + "/INCAR")
        else:
            sys.stderr.write("Error: In module mod_vasp.py'\n")
            sys.stderr.write(
                "       In subroutine 'setup_dfpt'\n")
            sys.stderr.write(
                "       Provide any of in_calc_fc, or src_dir/INCAR_DFPT " \
                "to be able to setup INCAR for dfpt calculations\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

        # Setup POTCAR
        if not potcar_created:
            if os.path.isfile("POTCAR"):
                copyfile("POTCAR", dest_dir + "/POTCAR")
            else:
                sys.stderr.write("Error: In module mod_vasp.py'\n")
                sys.stderr.write(
                    "       In subroutine 'setup_dfpt'\n")
                sys.stderr.write(
                    "       POTCAR file could not be created or found in the "\
                    "src_dir\n")
                sys.stderr.write("       Terminating!!!\n")
                exit(1)

        # KPOINTS (if KSPACING is present in INCAR, then do nothing)
        if not kpoints_created:
            if os.path.isfile("KPOINTS_DFPT"):
                copyfile("KPOINTS_DFPT", dest_dir + "/KPOINTS")
            elif os.path.isfile("KPOINTS"):
                copyfile("KPOINTS", dest_dir + "/KPOINTS")

    # Copy hostfile, this is only useful when relaxation and dfpt are run at the same time
    if hostfile is not None:
         copyfile(hostfile, dest_dir + "/" + hostfile)

    # Create magnetic moments file
    if in_atoms is not None:
        # Get Initial magnetic moments
        magmom_str = "MAGMOM = " + \
            " ".join(map(str, in_atoms.get_initial_magnetic_moments()))
        with open(dest_dir + "/magmom.conf", "w") as fh:
            fh.write(magmom_str)
    elif os.path.isfile("OUTCAR"):
        out_atoms = read_vasp_out("OUTCAR")
        # Get calculated magnetic moments
        magmom_str = "MAGMOM = " + \
            " ".join(map(str, out_atoms.get_magnetic_moments()))
        with open(dest_dir + "/magmom.conf", "w") as fh:
            fh.write(magmom_str)
    elif os.path.isfile("magmom.conf"):
        copyfile("magmom.conf", dest_dir + "/magmom.conf")
    else:
        pass

    # Decide supercell, Precedence: passed input super_cell > super_cell.conf > [2, 2, 2]
    # Precedence 1: Passed variable
    if super_cell is not None:
        if len(super_cell) != 3:
            sys.stderr.write("Error: In module mod_vasp.py'\n")
            sys.stderr.write(
                "       In subroutine 'setup_dfpt'\n")
            sys.stderr.write(
                "       'super_cell' variable should be a numpy array with 3 \
                integer elements\n")
            # Future: Print the passed super_cell array and its length
            sys.stderr.write("       Terminating!!!\n")
            exit(1)
    else:
        # Precedence 2: super_cell.conf file
        if os.path.isfile("super_cell.conf"):
            super_cell = mod_utility.read_array_data(
                "super_cell.conf", n_headers=0, separator=None, data_type="int")[0]
            if len(super_cell) != 3:
                sys.stderr.write("Error: In module mod_vasp.py'\n")
                sys.stderr.write("       In subroutine 'run_dfpt'\n")
                sys.stderr.write(
                    "       'super_cell.conf' file should contain 3 integers\n")
                # Future: Print the passed super_cell array and its length
                sys.stderr.write("       Terminating!!!\n")
                exit(1)
        # Precedence 3: Default
        else:
            super_cell = np.array([2, 2, 2])
    # Write super_cell.conf
    with open(dest_dir + "/super_cell.conf", 'w') as fhandle:
        fhandle.write(
            "%d %d %d" %
            (super_cell[0],
             super_cell[1],
             super_cell[2]))
             
             
    # Setup job driver exclusively for DFPT
    mod_batch.write_job_driver(work_dir=dest_dir)
    
    # Append DFPT processing instruction to the job driver in work_dir
    with open(dest_dir + '/' + job_driver_filename, 'a') as fh:
            fh.write("from matkit.calculators import mod_vasp\n")
            fh.write("mod_vasp.run_dfpt(work_dir='%s', n_proc=n_proc, " \
                     "hostfile=hostfile)\n" % (dest_dir))
                     
    # Move back to old directory
    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def run_dfpt(work_dir=None, super_cell=None, n_proc=1, hostfile=None):
    '''
    Runs a DFPT computation, then uses Phonopy to extract the force constants.
    Assumes the following files are present inside the work_dir.
    Necessary Files: INCAR
                     POSCAR-unitcell
                     POTCAR
                     KPOINTS
                     magmom.conf (optional)
                     super_cell.conf (optional)

    Typical contents of super_cell.conf file: Just one line Example: 2 2 2
    '''

    old_dir = os.getcwd()

    # If work directory is None, means run in the current directory
    if work_dir is not None:
        os.chdir(work_dir)

    # Check if all necessary files exist
    mod_utility.error_check_path_exists(
        pathname="INCAR",
        module="mod_vasp.py",
        subroutine="run_dfpt")
    mod_utility.error_check_path_exists(
        pathname="POSCAR-unitcell",
        module="mod_vasp.py",
        subroutine="run_dfpt")
    mod_utility.error_check_path_exists(
        pathname="POTCAR",
        module="mod_vasp.py",
        subroutine="run_dfpt")
    #mod_utility.error_check_path_exists(        # <<<<<<<<<<<<< KPOINTS need not exist if KSPACING is specified
    #    pathname="KPOINTS",
    #    module="mod_vasp.py",
    #    subroutine="run_dfpt")

    # Decide supercell
    # Precedence: passed input super_cell > super_cell.conf > [2, 2, 2]
    # Precedence 1: Passed variable
    if super_cell is not None:
        if len(super_cell) != 3:
            sys.stderr.write("Error: In module mod_vasp.py'\n")
            sys.stderr.write("       In subroutine 'run_dfpt'\n")
            sys.stderr.write(
                "       'super_cell' variable should be a numpy array with 3 \
                integer elements\n")
            # Future: Print the passed super_cell array and its length
            sys.stderr.write("       Terminating!!!\n")
            exit(1)
    else:
        # Precedence 2: super_cell.conf file
        if os.path.isfile("super_cell.conf"):
            super_cell = mod_utility.read_array_data(
                "super_cell.conf", n_headers=0, separator=None, data_type="int")[0]
            if len(super_cell) != 3:
                sys.stderr.write("Error: In module mod_vasp.py'\n")
                sys.stderr.write("       In subroutine 'run_dfpt'\n")
                sys.stderr.write(
                    "       'super_cell.conf' file should contain 3 integers\n")
                # Future: Print the passed super_cell array and its length
                sys.stderr.write("       Terminating!!!\n")
                exit(1)
        # Precedence 3: Default
        else:
            super_cell = np.array([2, 2, 2])
    super_cell_str = "'" + " ".join(map(str, super_cell)) + "'"

    # Create supercell for force constant computation
    if os.path.isfile("magmom.conf"):
        # If magnetic moments are present, then construct magnetic moments for
        # the super cell
        process = subprocess.call(
            "phonopy -d --dim=" +
            super_cell_str +
            " -c POSCAR-unitcell magmom.conf",
            shell=True)
    else:
        process = subprocess.call(
            "phonopy -d --dim=" +
            super_cell_str +
            " -c POSCAR-unitcell",
            shell=True)
    os.rename("SPOSCAR", "POSCAR")

    # If magnetic moments are present, modify INCAR
    if os.path.isfile("MAGMOM"):
        # The super cell generation command using phonopy would have generated
        # MAGMOM file
        magmom_data = mod_utility.read_array_data(
            "MAGMOM", n_headers=0, separator=None, data_type="string")[0]
        mod_utility.replace(
            file_path="INCAR",
            pattern="MAGMOM",
            subst=" ".join(magmom_data),
            replace_entire_line=True)
    else:
        mod_utility.replace(file_path="INCAR", pattern="MAGMOM", subst='',
            replace_entire_line=True)
    
    # Run vasp to compute force constants
    run_vasp(n_proc=n_proc, hostfile=hostfile)

    # Extract force constants from vasprun.xml file
    process = subprocess.call("phonopy --fc vasprun.xml", shell=True)

    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def setup_stress_constraint_file(work_dir, mode_id, stress_voigt, stress_filename):

    '''
    NOTE: Deprecated. This is an older version, no longer supported.
    Will not setup the file if mode_id is 0
    '''

    if mode_id == 0:
        return

    mod_utility.error_check_path_exists(
            pathname=work_dir, module=module_name,
            subroutine="setup_stress_constraint_file")
    old_dir = os.getcwd()
    os.chdir(work_dir)

    with open(stress_filename, 'w') as fh:
        fh.write("%d\n" %(mode_id))
        # XX   YY   ZZ   XY   YZ   ZX, which is pressure, -stress, in kB.
        fh.write("%f %f %f %f %f %f" %(-10.0*stress_voigt[0], \
                 -10.0*stress_voigt[1], -10.0*stress_voigt[2], \
                 -10.0*stress_voigt[5], -10.0*stress_voigt[3], \
                 -10.0*stress_voigt[4]))

    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def setup_constrained_relaxation_file(work_dir, mode_id, flag_voigt_list,
        value_voigt_list, constr_setup_filename='CONSTR_CELL_RELAX_SETUP'):

    '''
    flags_voigt: A list of 
    '''
    if (mode_id == 0) or (flag_voigt_list is None) or (value_voigt_list is None):
        return
        
    mod_utility.error_check_path_exists(
            pathname=work_dir, module=module_name,
            subroutine="setup_constrained_relaxation_file")
    old_dir = os.getcwd()
    os.chdir(work_dir)

    with open(constr_setup_filename, 'w') as fh:

        fh.write("%d\n" %(mode_id))
        # Flags in voigt order
        fh.write("%s %s %s %s %s %s\n" %(flag_voigt_list[0], flag_voigt_list[1], \
                 flag_voigt_list[2], flag_voigt_list[3], flag_voigt_list[4], \
                 flag_voigt_list[5]))

        # Values in voigt order
        fh.write("%f %f %f %f %f %f" %(value_voigt_list[0], value_voigt_list[1], \
                 value_voigt_list[2], value_voigt_list[3], value_voigt_list[4], \
                 value_voigt_list[5]))

    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def veryclean( ):

    file_list = dist_clean_file_list
    clean(file_list=file_list)


##################
### SUBROUTINE ###
##################


def clean(file_list=None):

    if file_list is None:

        file_list = basic_clean_file_list

    for f in file_list:
        try:
            os.remove(f)
        except OSError:
            pass

##################
### SUBROUTINE ###
##################


def tree_clean(work_dir="./", file_list=None, clean_level="basic"):

    '''
    Clean up after vasp computation. Only cleans "unnecessary" files.
    '''

    # Set file list to clean if not specified by the user
    if file_list is None:

        mod_utility.error_check_argument_required(
            arg_val=clean_level, arg_name="clean_level", module=module_name,
            subroutine="tree_clean", valid_args=["basic", "dist", "saver"])

        if clean_level == "basic":
            file_list = basic_clean_file_list

        if clean_level == "dist":
            file_list = dist_clean_file_list

        if clean_level == "saver":
            file_list = saver_clean_file_list

    # For each file type
    for f in file_list:

        # Search the directory tree
        for parent, dirnames, filenames in os.walk(work_dir):

            # Delete each occurance in the tree
            for fn in filenames:
                if fn == f:
                    try:
                        os.remove(os.path.join(parent, fn))
                    except OSError:
                        pass

'''----------------------------------------------------------------------------
                                    END OF MODULE
----------------------------------------------------------------------------'''
