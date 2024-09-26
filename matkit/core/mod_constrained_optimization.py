'''-----------------------------------------------------------------------------
                        mod_constrained_optimization.py

 Description: This module performs a constrained optimization of supercell and
   internal (ionic) degrees of freedom with applied constraints of stress and/or 
   fixed (strain) on appropriate coordinate directions. In general most DFT
   codes only allow a very restricted set of relaxations or geometry
   optimization methods. This module should serve as a general procedure to
   optimize a supercell (and internal coordinates) particularly for use in
   theoretical strength calculations and in conjucntion with any
   atomistic/continuum method, that can compute stress (Currently only VASP is
   supported).
   
   NOTE: For itermation_method the default 'elasticity' is currently reliable.
         'linear_regression' and 'random_forest_regression' are experimental.
   
   IMPORTANT: If using copt_inbuilt, ISIF should be 2
              If using copt_vasp , ISIF should be 3

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json
import time
import shutil
import string
import inspect
import warnings
import subprocess
import numpy as np
from typing import List, Union, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Externally installed modules
from ase.io.vasp import read_vasp, write_vasp

# Local imports
from matkit import CONFIG
from matkit.core import mod_tensor, mod_utility, mod_batch
from matkit.calculators import mod_calculator, mod_vasp

'''-----------------------------------------------------------------------------
                                     MODULE VARIABLES
-----------------------------------------------------------------------------'''
module_name = 'mod_constrained_optimization.py'
COPT_INPUT_FILENAME='COPT_INPUT'
COPT_DRIVER_FILENAME='COPT_DRIVER'

# What VASP files can be deleted safely
copt_clean_file_list = ['CHG', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 
    'XDATCAR', 'PROCAR', 'REPORT', 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2',
    'ase-sort.dat', 'OSZICAR', 'vasprun.xml', 'POTCAR', '#DONE#',
    'console.report.vasp', 'WAVECAR', 'CHGCAR']

template_copt_input = {
    'mode_id' : 1, # int
    'flags_voigt' : None, # List[str]
    'target_stress_voigt' : None, # List[Union[int, float]]
    'optimization_engine' : 'inbuilt_copt', #Optional[List[Union[int, float]]]
    'stress_conv_tols_voigt' : None,
    'max_iterations' : None, # Optional[int],
    'iteration_method' : None, # Optional[str]
    'E' : None, # Union[int, float]
    'Nu' : None, # float
    'G' : None, # Calculated based on E and Nu
}


'''-----------------------------------------------------------------------------
                              MODULE CLASSES AND SUBROUTINES
-----------------------------------------------------------------------------'''

##################################
### CLASS TO HANDLE COPT INPUT ###
##################################


def write_constrained_optimization_input(copt_input_dict, copt_input_file_path: str, is_verbose: bool = False):

    '''
    This subroutine is common for both VASP based patch for constrained cell
    optimization (not publicly released due to VASP liscence limitation) and the
    generic constrained cell optimization implemented in this module that can
    work with any code that can compute stresses.
    
    The input file for constrained optimization has 3 lines in the following format:
    <modeid>
    <6 character flags separated by spaces in voigt order: S => stress control F => Fixed cell>
    <6 numeric (integer or double precision) values separated by spaces corresponding to above the flags in line 2>
    
    Explanation:
    LINE 1: modeid should be integer, currently only 1 is accepted, and is the
            most general setting, where all 6 components of the cell can be
            controlled (either stress or fixed).
    LINE 2: Example: F F S F F F, this will fix XX YY YZ ZX XY directions and
            control stress in ZZ direction to the desired value.
    LINE 3: Six numerical values. Values corresponding to 'F' flags are dymmy
            but need to be numerical, so you can set them to be 0 (for example).
            Values corresponding to flag S are stresses, in the units of GPa
            Sign convention of stress is what is normally used (+ for applied
            tension, - for applied compression etc)
    Both the flags as well as the stress/strain parameters should have Voigt
    ordering, explained below:
        1 = > (1,1) => (X,X)
        2 = > (2,2) => (Y,Y)
        3 = > (3,3) => (Z,Z)
        4 = > (2,3) => (Y,Z)
        5 = > (1,3) => (X,Z)
        6 = > (1,2) => (X,Y)
    '''

    # Check validity of input data
    copt = copt_input_dict.copy()
    sub_name = inspect.currentframe().f_code.co_name

    if copt['mode_id'] != 1:
        raise ValueError(f"In {sub_name}: mode_id (currently) can only accept an integer value of 1 corresponding to general constrained optimization method")
            
    if len(copt['flags_voigt']) != 6:
        raise ValueError(f"In {sub_name}: flags_voigt must be a character list of length 6")
    elif not all(x in ['S', 'F'] for x in copt['flags_voigt']):
        raise ValueError(f"In {sub_name}: All elements in the flags_voigt list must be either 'S' or 'F' representing stress or fixed constraints")

    if len(copt['target_stress_voigt']) != 6:
        raise ValueError(f"In {sub_name}: target_stress_voigt must be a numeric list of length 6")
    elif not all(isinstance(x, (int, float)) for x in copt['target_stress_voigt']):
        raise ValueError(f"In {sub_name}: All elements in the target_stress_voigt list must be of type int or float")
        
    if copt['optimization_engine'] not in ['inbuilt_copt', 'vasp_copt']:
        raise ValueError(f"In {sub_name}: Unknown optimization_engine, allowed engines are inbuilt_copt or vasp_copt")

    # Write to file
    dir_name = os.path.dirname(copt_input_file_path)
    if dir_name and not os.path.exists(dir_name):
        raise FileNotFoundError(f"In {sub_name}: The directory '{dir_name}' does not exist")
    elif not os.access(dir_name, os.W_OK):
        raise PermissionError(f"In {sub_name}: Write permission is denied for the directory '{dir_name}'")


    if copt['optimization_engine'] == 'vasp_copt':
        # Stress converegce is indirectly controlled through EDIFFG in VASP COPT Patch
        with open(copt_input_file_path, 'w') as fh:
            fh.write(f"{copt['mode_id']}\n")
            fh.write(" ".join(copt['flags_voigt']) + "\n")
            fh.write(" ".join(str(x) for x in copt['target_stress_voigt']) + "\n")

   
    elif copt['optimization_engine'] == 'inbuilt_copt':
    
        # inbuilt_copt check the validity of the rest of the inputs
        if copt['stress_conv_tols_voigt'] is None:
            if is_verbose:
                warnings.warn(f"In {sub_name}: stress_conv_tols_voigt is not provided, using the default 0.1 * np.ones(6) GPa", UserWarning)
            copt['stress_conv_tols_voigt'] = 0.1 * np.ones(6)
        elif len(copt['stress_conv_tols_voigt']) != 6:
            raise ValueError(f"In {sub_name}: stress_conv_tols_voigt must be a numeric list of length 6")
        elif not all(isinstance(x, (int, float)) for x in copt['stress_conv_tols_voigt']):
            raise ValueError(f"In {sub_name}: All elements in the stress_conv_tols_voigt list must be of type int or float")

        if copt['max_iterations'] is None:
            if is_verbose:
                warnings.warn(f"In {sub_name}: max_iterations is not provided, using a default value of 50", UserWarning)
            copt['max_iterations'] = 50
        elif not isinstance(copt['max_iterations'], (int)):
            raise ValueError(f"In {sub_name}: max_interations should be an integer")
                
        if copt['iteration_method'] is None:
            if is_verbose:
                warnings.warn(f"In {sub_name}: iteration_method is not provided, using default elasticity", UserWarning)
            copt['iteration_method'] = 'elasticity'
        elif copt['iteration_method'] not in ['elasticity', 'linear_regression', 'random_forest_regression']:
            raise ValueError(f"In {sub_name}: Unknown method, allowed methods are elasticity, linear_regression and random_forest_regression")
 
        # Read elastic parameters for first iteration
        if copt['E'] is None:
            if is_verbose:
                warnings.warn(f"In {sub_name}: E is not provided, using a default value of 100.0 GPa", UserWarning)
            copt['E'] = 100.0 #165 #200.0
        elif not isinstance(copt['E'], (int, float)):
            raise ValueError(f"In {sub_name}: E should be an integer or a float")

        if copt['Nu'] is None:
            if is_verbose:
                warnings.warn(f"In {sub_name}: Nu is not provided, using a default value of 0.3", UserWarning)
            copt['Nu'] = 0.3 #0.22 #0.3
        elif not isinstance(copt['Nu'], (float)):
            raise ValueError(f"In {sub_name}: Nu should be a  float between 0.0 and 0.5 typically between 0.2 and 0.3 for most solids")
            
        # Shear modulus in GPa
        copt['G'] = copt['E'] / (2-2*copt['Nu'])
        
        with open(copt_input_file_path, 'w') as fh:
            json.dump(copt, fh, indent=4, cls=mod_utility.json_numpy_encoder)
  
###################################
### CLASS TO HANDLE COPT DRIVER ###
###################################


class Constrained_Optimization_Driver:

    '''
    The driver class for constrained optimization.
    The class data is a dictionary with the following fields:
        1. input_dict: Constrained optimization input settings
        3. stress_or_fixed_voigt: Processed constrained optimization settings, values are 0 if fixed, 1 if stress controlled
        4. Trajectory dictionary array. Each point on the trajectory is a dictionary with fields:
            a. supercell : A 3x3 matrix describing the supercell in Angstroms. Each row is a supercell vector.
            b. stress_voigt: Stress corresponding to the supercell, in voight notation and in GPa.
            NOTE: If stress_voigt is None, that means the calculation is not yet done or failed.
    '''

    ##################
    ### SUBROUTINE ###
    ##################
    def __init__(self, driver_file_path: str, input_file_path: str = None, reference_supercell: np.ndarray = None):

        self.driver_file_path = driver_file_path

        # Restart mode
        if os.path.exists(driver_file_path):
            self.read_driver_from_file()
                
        # Fresh mode
        else:
        
            self.data = {
                'input_dict' : None,
                'stress_or_fixed_voigt' : None,
                'trajectory': [],
                'is_stop': False,
                'is_converged' : False,
                'last_successful_iter_idx' : -1 # Start numbering from 0 for reference supercell optimization
            }

            # Add reference supercell
            self.push_next_supercell_to_trajectory(reference_supercell)
            
            # Read Constrained optimization input file
            if os.path.isfile(input_file_path):
                with open(input_file_path, 'r') as fh:
                    self.data['input_dict'] = json.load(fh)
                
                self.data['stress_or_fixed_voigt'] = np.zeros(6, dtype=int)
                for voigt_idx, flag in enumerate(self.data['input_dict']['flags_voigt']):
                    if flag == 'S':
                        self.data['stress_or_fixed_voigt'][voigt_idx] = 1
                        
            self.write_driver_to_file()
            
    ##################
    ### SUBROUTINE ###
    ##################
    def read_driver_from_file(self):
        with open(self.driver_file_path, 'r') as fh:
            self.data = json.load(fh)

    ##################
    ### SUBROUTINE ###
    ##################
    def write_driver_to_file(self):
        with open(self.driver_file_path, 'w') as fh:
            json.dump(self.data, fh, indent=4, cls=mod_utility.json_numpy_encoder)

    ##################
    ### SUBROUTINE ###
    ##################
    def push_next_supercell_to_trajectory(self, supercell: np.ndarray):

        method_name = inspect.currentframe().f_code.co_name
        
        if supercell.shape != (3,3):
            raise ValueError(f"In {self.__class__.__name__}.{method_name}: supercell must be 3x3 numpy.ndarray")
        
        if np.linalg.matrix_rank(np.array(supercell)) < 3:
            raise ValueError(f"In {self.__class__.__name__}.{method_name}: supercell is rank deficient")
            
        # Check if the last calculation is complete before pushing the next supercell
        if len(self.data['trajectory']) > 0:
            if self.data['trajectory'][-1]['stress_voigt'] is None:
                raise ValueError(f"In {self.__class__.__name__}.{method_name}: Trying to push the next supercell to trajectory without computing the stress in the previous iteration")

        entry = {
            'supercell': supercell.tolist(),
            'stress_voigt': None,
        }
        self.data['trajectory'].append(entry)
        self.write_driver_to_file()

    ##################
    ### SUBROUTINE ###
    ##################    
    def push_computed_stress_to_trajectory(self, stress_voigt: np.array):
    
        method_name = inspect.currentframe().f_code.co_name
    
        if len(stress_voigt) != 6:
            raise ValueError(f"In {self.__class__.__name__}.{method_name}: stress_voigt must be a length 6 numpy.array")
            
        # Check trajectory before pushing the stress
        if len(self.data['trajectory']) == 0:
            raise ValueError(f"In {self.__class__.__name__}.{method_name}: Trying to push stress to an empty trajectory")
        else:
            if self.data['trajectory'][-1]['stress_voigt'] is not None:
                raise ValueError(f"In {self.__class__.__name__}.{method_name}: Trying to push stress to trajectory of already finished calculation")

            self.data['trajectory'][-1]['stress_voigt'] = stress_voigt.copy()
            
            # Update iteration counter
            self.data['last_successful_iter_idx'] = self.data['last_successful_iter_idx'] + 1
            
            # Check termination criterion
            if self.data['last_successful_iter_idx'] >= self.data['input_dict']['max_iterations']:
                self.data['is_stop'] = True
            
            if self.is_converged():
                self.data['is_stop'] = True
                self.data['is_converged'] = True
                
            self.write_driver_to_file()

    ##################
    ### SUBROUTINE ###
    ##################    
    def is_converged(self):
    
        # Target stress of non fixed components
        target_stress_voigt = np.array(self.data['input_dict']['target_stress_voigt'].copy())
        target_stress_voigt = np.multiply(target_stress_voigt, self.data['stress_or_fixed_voigt'])
        
        # Current stress of non-fixed components
        current_stress_voigt = np.array(self.data['trajectory'][-1]['stress_voigt'].copy())
        current_stress_voigt = np.multiply(current_stress_voigt, self.data['stress_or_fixed_voigt'])
        
        diff_stress_voigt = np.absolute(target_stress_voigt - current_stress_voigt)
        stress_conv_tols_voigt = np.array(self.data['input_dict']['stress_conv_tols_voigt'])
        
        return (diff_stress_voigt <= stress_conv_tols_voigt).all()

    ##################
    ### SUBROUTINE ###
    ##################    
    def get_stress_strain_voigt_history(self):

        stress_voigt_list = []
        strain_voigt_list = []

        ref_supercell = self.data['trajectory'][0]['supercell'].copy()
        inv_ref_supercell = np.linalg.inv(ref_supercell)
        
        for idx in range(len(self.data['trajectory'])):

            stress_voigt_list.append(self.data['trajectory'][idx]['stress_voigt'].copy())
            supercell = self.data['trajectory'][idx]['supercell'].copy()
            F = np.matmul(supercell, inv_ref_supercell)
            E = mod_tensor.green_lagrange_strain_tensor(F)
            E_voigt = mod_tensor.full_3x3_to_voigt_6_strain(strain_matrix=E)
            
            # Make sure that the fixed components are exactly zero
            E_voigt = np.multiply(E_voigt, self.data['stress_or_fixed_voigt'])
            strain_voigt_list.append(E_voigt.copy())
        return (stress_voigt_list, strain_voigt_list)

    ##################
    ### SUBROUTINE ###
    ##################
    def get_next_supercell(self):
    
        '''
        Predicts the incremental deformation to be applied to the last converged
        supercell to obtain the target stress
        '''
    
        # First iteration is not done yet (reference supercell stress not yet computed)
        if len(self.data['trajectory']) == 1 and self.data['trajectory'][0]['stress_voigt'] is None:
            return self.data['trajectory'][0]['supercell']
        
        # First iteration is done (reference supercell stress available)
        elif self.data['input_dict']['iteration_method'] == 'elasticity' or len(self.data['trajectory']) == 1:

            # Stress increment needed (target stress - current stress)  
            dsigma_voigt = np.array(self.data['input_dict']['target_stress_voigt']) - np.array(self.data['trajectory'][-1]['stress_voigt'])
            
            # Make increments of target stress on fixed components to be zero
            dsigma_voigt = np.multiply(dsigma_voigt, self.data['stress_or_fixed_voigt'])
            
            # Strain increment to acheive the target stress
            # FUTURE: Implement bounds on the maximum strain (step size control)
            deps_voigt = np.zeros(6)
            deps_voigt[0] = ( dsigma_voigt[0] - self.data['input_dict']['Nu'] * (dsigma_voigt[1] + dsigma_voigt[2]) ) / self.data['input_dict']['E']
            deps_voigt[1] = ( dsigma_voigt[1] - self.data['input_dict']['Nu'] * (dsigma_voigt[2] + dsigma_voigt[0]) ) / self.data['input_dict']['E']
            deps_voigt[2] = ( dsigma_voigt[2] - self.data['input_dict']['Nu'] * (dsigma_voigt[0] + dsigma_voigt[1]) ) / self.data['input_dict']['E']
            
            # Recall 0.5 factor is present for full strain tensor notation as opposed to voigt notation
            deps_voigt[3] = dsigma_voigt[3] / self.data['input_dict']['G']
            deps_voigt[4] = dsigma_voigt[4] / self.data['input_dict']['G']
            deps_voigt[5] = dsigma_voigt[5] / self.data['input_dict']['G']
            
            # Make increments of strain on fixed components to be zero
            deps_voigt = np.multiply(deps_voigt, self.data['stress_or_fixed_voigt'])
            deps = mod_tensor.voigt_6_to_full_3x3_strain(deps_voigt)
           
            #incremental_right_stretch = np.diag([1,1,1]) + deps # In future multiply by an acceleration factor to reduce fluctuations and achieve smooth convergence
            incremental_F = mod_tensor.convert_strain_to_deformation(deps)
            
            next_supercell = mod_tensor.apply_defgrad_to_cell(incremental_F, self.data['trajectory'][-1]['supercell'])
            self.push_next_supercell_to_trajectory(next_supercell)
            
            return next_supercell

        elif self.data['input_dict']['iteration_method'] == 'linear_regression' or self.data['input_dict']['iteration_method'] == 'random_forest_regression':
        
            if self.data['input_dict']['iteration_method'] == 'linear_regression':
                model = LinearRegression()
            elif self.data['input_dict']['iteration_method'] == 'random_forest_regression':
                model = RandomForestRegressor(n_estimators=100, random_state=0)
                
            [stress_voigt_list, strain_voigt_list] = self.get_stress_strain_voigt_history()
            model.fit(stress_voigt_list, strain_voigt_list)
            
            # Get target Green-Lagrange (finite) strain tensor
            target_strain_voigt = model.predict([ np.array(self.data['input_dict']['target_stress_voigt']) ])[0]
            
            # Make sure that the predicted strain respectes the fixed constraints
            target_strain_voigt = np.multiply(target_strain_voigt, self.data['stress_or_fixed_voigt'])         
            target_strain = mod_tensor.voigt_6_to_full_3x3_strain(target_strain_voigt)
            
            # Compute the deformation gradient (w.r.t reference supercell) and compute the new supercell
            F = mod_tensor.convert_strain_to_deformation(target_strain)
            
            next_supercell = mod_tensor.apply_defgrad_to_cell(F, self.data['trajectory'][0]['supercell'])
            
            self.push_next_supercell_to_trajectory(next_supercell)
            return next_supercell

'''-----------------------------------------------------------------------------
                                 SUBROUTINES
-----------------------------------------------------------------------------'''
  
##################
### SUBROUTINE ###
##################


def setup(work_dir, sim_info, structure=None, calculator_input=None, batch_object=None):

    '''
    This subroutine sets up the basic inputs needed to run a contrained optimization
    '''
    
    # Create and move to work directory
    old_dir = os.getcwd()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
  
    if (structure is not None) and (calculator_input is not None):
    
        if sim_info['calculator'] == 'vasp':
            if sim_info['sim_specific_input']['optimization_engine'] == 'inbuilt_copt':
                sim_info['sim_type'] = 'relax_i'
            elif sim_info['sim_specific_input']['optimization_engine'] == 'vasp_copt':
                sim_info['sim_type'] = 'relax_isv'

        # Create the DFT calculation files
        mod_calculator.run_or_prepare_sim(
            calculator=sim_info['calculator'],
            in_atoms=structure, in_calc=calculator_input,
            n_proc=sim_info['n_proc'], sim_type=sim_info['sim_type'],
            n_iter=sim_info['n_relax_iter'], is_prepare=True,
            final_static=sim_info['is_final_static'],
            is_direct_coordinates=sim_info['is_direct_coordinates'])

        mod_utility.delete_files(file_list=['ase-sort.dat'])
   
    # Create constrained optimization input file in the work_dir
    write_constrained_optimization_input(sim_info['sim_specific_input'], \
        copt_input_file_path="./" + COPT_INPUT_FILENAME)

    if batch_object is not None:

        # Write a submit script (only the header)
        batch_object.ntasks = sim_info['n_proc']
        mod_batch.generate_submit_script(batch_object=batch_object)

        # Append job_driver.py processing call to the slurm file
        with open(batch_object.file_name, 'a') as fh:
            fh.write("python job_driver.py %s\n"  % (mod_batch.n_proc_str()))

        # Create job_driver.py (header)
        mod_batch.write_job_driver(work_dir="./")

        # Append commands specific to sim_type to job_driver.py
        with open("job_driver.py", "a") as fh:
            fh.write("from matkit.core import mod_constrained_optimization\n"
                     "mod_constrained_optimization.run"\
                     "(work_dir='%s', n_proc=n_proc, hostfile=hostfile,"\
                     " parset_filename=None)\n"
                     %(work_dir))

    # Add a waiting tag
    open("#WAITING#", 'a').close()

    # Move back to where you were
    os.chdir(old_dir)
    
#################
### SUBROUTINE ###
##################


def run(work_dir, n_proc, hostfile=None, parset_filename=None, save_level=0):

    '''
    Runs constrained optimization in the work directory
    save_level = 0 : Only contents of the final "converged" result directory, copied to work_dir
                 1 : Leave as it is
    '''

    # Create and move to work directory
    old_dir = os.getcwd()
    os.chdir(work_dir)
            
    # Check if there is a #WAITING# tag file
    if os.path.isfile("#WAITING#"):
        os.rename("#WAITING#", "#PROCESSING#")
    else:
        return False

    # Read the reference_supercell
    in_atoms = mod_calculator.read_inputs_get_ase(calculator='vasp', work_dir="./")
    reference_supercell = in_atoms.cell[:]
    
    driver = Constrained_Optimization_Driver(
        driver_file_path=COPT_DRIVER_FILENAME,
        input_file_path=COPT_INPUT_FILENAME,
        reference_supercell=reference_supercell
    )
    
    while not driver.data['is_stop']:
    
        # Get atoms
        if driver.data['last_successful_iter_idx'] == -1:
            prev_iter_dir = "./"
            curr_atoms = mod_calculator.read_inputs_get_ase(calculator='vasp', work_dir="./")
        else:
            curr_supercell = driver.get_next_supercell()
            prev_iter_dir = "ITER_" + str(driver.data['last_successful_iter_idx']).zfill(3)
            prev_atoms = read_vasp(prev_iter_dir + '/CONTCAR')
            #curr_atoms = mod_ase.defgrad_ase_atoms(in_atoms=prev_atoms, F=F)
            prev_atoms.set_cell(curr_supercell, scale_atoms=True)
            curr_atoms = prev_atoms
       
        # If current directory exists, remove it and create new one (should be done for restart calulcations)
        curr_iter_dir = "ITER_" + str(driver.data['last_successful_iter_idx']+1).zfill(3)
        if os.path.isdir(curr_iter_dir):
            shutil.rmtree(curr_iter_dir)
        os.mkdir(curr_iter_dir)
        
        # Copy INCAR, POTCAR from reference to current
        shutil.copyfile('./INCAR', curr_iter_dir + '/INCAR')
        shutil.copyfile('./POTCAR', curr_iter_dir + '/POTCAR')

        # WAVECAR has no use expect for starting next continuation job, so move it altogether to current directory from previous if present
        if os.path.isfile(prev_iter_dir + '/WAVECAR'):
            shutil.move(prev_iter_dir + '/WAVECAR', curr_iter_dir + '/WAVECAR')            

        # CHGCAR has further use at each iteration for plotiing charge densities, so copy it to current from previous
        if os.path.isfile(prev_iter_dir + '/CHGCAR'):
            shutil.copyfile(prev_iter_dir + '/CHGCAR', curr_iter_dir + '/CHGCAR')
                    
        # Create a waiting file
        open(curr_iter_dir + '/#WAITING#', 'a').close()

        # Copy hostfile if needed. If it exists, it is in the work_dir
        if hostfile is not None:
            shutil.copyfile(hostfile, curr_iter_dir + '/' + hostfile)

        # Move to the current iteration directory
        os.chdir(curr_iter_dir)

        # Create POSCAR in the current directory
        write_vasp(file='POSCAR', atoms=curr_atoms)

        # Run DFT job
        time.sleep(CONFIG.SLEEP_TIME)
        curr_dir_abs_path = os.getcwd()
        mod_calculator.run_dft_job(calculator='vasp', n_proc=n_proc, n_iter=1,
            is_final_static=False, hostfile=hostfile,
            wdir=curr_dir_abs_path, parset_filename=parset_filename)
            
        # Clean up the current directory, for space storage
        mod_vasp.clean(file_list=copt_clean_file_list)
        
        # Check convergence
        # NOTE: If a run fails to converge delete the directory
        # Always read relaxation results
        with open('dft_results_0.json') as fh:
            dft_results_0 = json.load(fh)
        stress_voigt_current=np.array(dft_results_0['stress_voigt_GPa'])
        
        # Move to work_dir
        os.chdir(work_dir)

        # Push stress NOTE: Be in current directory, so that driver saves to file properly
        driver.push_computed_stress_to_trajectory(stress_voigt_current)
        
    # Clean up
    if save_level == 0:
        # Remove intermediate iterations
        for iter_idx in range(0, driver.data['last_successful_iter_idx']):
            iter_dir = "ITER_" + str(iter_idx).zfill(3)
            shutil.rmtree(iter_dir)
        # Move the final iteration contents (CONTCAR, WAVECAR, CHGCAR, dft_results_0.json) to work directory
        if driver.data['is_converged']:
            src_dir = "ITER_" + str(driver.data['last_successful_iter_idx']).zfill(3)
            for flnm in ['CONTCAR', 'OUTCAR', 'dft_results_0.json']:
                shutil.copyfile(src_dir + '/' + flnm, './' + flnm)
            shutil.rmtree(src_dir)
            os.rename("#PROCESSING#", "#DONE#")
            mod_vasp.clean(file_list=copt_clean_file_list)


    # Move back to where you were
    os.chdir(old_dir)
    
    return driver.data['is_converged']

'''-----------------------------------------------------------------------------
                                END OF MODULE
-----------------------------------------------------------------------------'''
