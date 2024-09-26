'''----------------------------------------------------------------------------
                             mod_dft.py

 Description: Some basic utilities for DFT based claculations.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import json
from pathlib import Path
from copy import deepcopy

# Externally installed modules
# None

# Local imports
from matkit.calculators import mod_calculator
from matkit.core import mod_batch, mod_utility, mod_constrained_optimization

'''----------------------------------------------------------------------------
                                     MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_dft.py"

##################
### SUBROUTINE ###
##################


def setup_basic_dft_calculations(
        work_dir, rel_dir_list, structure_list, calculator_input_list,
        sim_info_list, batch_object, is_taskfarm=False, n_taskfarm=1,
        taskfarm_n_proc=-1):
        
    '''
    With in a given work_dir, setsup a series of DFT calculations with
    subdirectory given by rel_dir_list.
    '''

    # Create work dir and move to it
    old_dir = os.getcwd()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    work_dir = os.getcwd()

    # Write directory list to the work directory
    calc_info = {
        'rel_dir_list': rel_dir_list
    }
    with open('calc_info.json', 'w') as fh:
        json.dump(calc_info, fh, indent=4, cls=mod_utility.json_numpy_encoder)

    for idx, (rel_dir, structure, calculator_input, sim_info) in enumerate(zip(rel_dir_list, structure_list, calculator_input_list, sim_info_list), start=0):

        # Create and move to local_work dir
        Path(rel_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(rel_dir)
        
        # Create the DFT calculation files
        mod_calculator.run_or_prepare_sim(
            calculator=sim_info['calculator'],
            in_atoms=structure, in_calc=calculator_input,
            n_proc=sim_info['n_proc'], sim_type=sim_info['sim_type'],
            n_iter=sim_info['n_relax_iter'], is_prepare=True,
            final_static=sim_info['is_final_static'],
            is_direct_coordinates=sim_info['is_direct_coordinates'])

        # Write a submit script (only the header)
        batch_object_local = deepcopy(batch_object)
        batch_object_local.ntasks = sim_info['n_proc']
        mod_batch.generate_submit_script(batch_object=batch_object_local)

        # Append job_driver.py processing call to the slurm file
        with open(batch_object.file_name, 'a') as fh:
            fh.write("python job_driver.py %d\n"  % (sim_info['n_proc']))

        # Create job_driver.py (header)
        mod_batch.write_job_driver(work_dir="./")

        # Append commands specific to sim_type to job_driver.py
        with open("job_driver.py", "a") as fh:
            fh.write("from matkit.calculators import mod_calculator\n"
                     "mod_calculator.run_dft_job"\
                     "(calculator='%s', n_proc=n_proc, is_final_static=%s, hostfile=hostfile,"\
                     " parset_filename=None, n_iter=%d, wdir='%s')\n"
                     %(sim_info['calculator'], sim_info['is_final_static'], sim_info['n_relax_iter'], work_dir+'/'+rel_dir))

        # Add a waiting tag
        open("#WAITING#", 'a').close()
        
        # Store some necessary information into the rel_dir folder relating to the job
        job_info = {
            'calculator' : sim_info['calculator'],
            'n_proc' : sim_info['n_proc']
        }
        with open('job_info.json', 'w') as fh:
            json.dump(job_info, fh, indent=4, cls=mod_utility.json_numpy_encoder)

        #############################
        # SIMULATION SPECIFIC SETUP #
        #############################
        
        # If constrained cell relaxation
        if sim_info['sim_type'] == 'copt':
            mod_constrained_optimization.setup(work_dir="./", sim_info=sim_info)

        os.chdir(work_dir)

    # Create a task farm file
    if is_taskfarm:
        mod_batch.create_taskfarm(
            batch_object=batch_object,
            n_taskfarm=n_taskfarm,
            taskfarm_n_proc=taskfarm_n_proc,
            abs_job_dir_list=[work_dir + '/' + rel_dir for rel_dir in rel_dir_list],
            work_dir=work_dir, job_driver_filename="job_driver.py")

    # Move back to old directory
    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def process_basic_dft_calculations(work_dir, is_silent=True):

    # Move to work directory
    mod_utility.error_check_dir_exists(dirname=work_dir, module=module_name, \
        subroutine="process_basic_dft_calculations")
    old_dir = os.getcwd()
    os.chdir(work_dir)

    # Load the calc_info.json
    with open("calc_info.json") as fh:
        calc_info = json.load(fh)

    # Results list of dicts and None if no dft results found
    calc_results = {
        "dft_results_0_list": [],
        "dft_results_1_list": []
    }

    # Process each directory
    for rel_dir in calc_info['rel_dir_list']:

        file_0 = rel_dir + "/dft_results_0.json"
        file_1 = rel_dir + "/dft_results_1.json"

        is_file_0 = os.path.isfile(file_0)
        is_file_1 = os.path.isfile(file_1)

        if is_file_0:
            with open(file_0) as fh:
                dft_results_0 = json.load(fh)
            calc_results["dft_results_0_list"].append(dft_results_0)
        else:
            calc_results["dft_results_0_list"].append(None)

        if is_file_1:
            with open(file_1) as fh:
                dft_results_1 = json.load(fh)
            calc_results["dft_results_1_list"].append(dft_results_1)
        else:
            calc_results["dft_results_1_list"].append(None)

        # Warn if no dft results are present
        if (not is_file_0) and (not is_file_1) and (not is_silent):
            warnings.warn(("In module '%s'\n"
                           "In subroutine 'process_basic_dft_calculations'\n"
                           "DFT results missing in directory: '%s'"
                            %(module_name, work_dir+"/"+rel_dir)), stacklevel=3)

    # Save the results to work_dir (already located in work_dir)
    with open('calc_results.json', 'w') as fhandle:
        json.dump(calc_results, fhandle, indent=4,
                  cls=mod_utility.json_numpy_encoder)

    # Move back to old directory
    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def extract_basic_dft_calculations(work_dir, is_silent=True):

    # Move to work directory
    mod_utility.error_check_dir_exists(dirname=work_dir, module=module_name, \
        subroutine="extract_basic_dft_calculations")
    old_dir = os.getcwd()
    os.chdir(work_dir)

    # Load the calc_info.json
    with open("calc_info.json") as fh:
        calc_info = json.load(fh)

    # Extract DFT results in each directory
    for rel_dir in calc_info['rel_dir_list']:

        os.chdir(rel_dir)
        mod_calculator.extract_dft_job(calc_info['calculator'], is_final_static=False, wdir=None)
        os.chdir(work_dir)

    # Move back to old directory
    os.chdir(old_dir)

'''----------------------------------------------------------------------------
                               END OF MODULE
----------------------------------------------------------------------------'''
