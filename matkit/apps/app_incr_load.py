'''-----------------------------------------------------------------------------
                               app_incr_load.py

  Description: Performs an incremental loading simulation for theoretical/ideal
  strength calculations.

  Features:
    1. Ideal strength by affine straining with stress/strain boundary conditions
    2. Phonon stability analysis along the affine strain path
    3. AIMD calculations of ideal strength
     
    NOTE: In the affine straining and resulting phonon instability case, one
          can also impose random disturbance on the degrees of freedom to break
          the symmetry (But this is not currently implemented).

    NOTE: AIMD calculations use a larger supercell compared to affine
          calculations.

  Author: Subrahmanyam Pattamatta
  Contact: lalithasubrahmanyam@gmail.com
-----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import json
import argparse
import importlib

# Externally installed modules
# None

# Local imports
from matkit import CONFIG
from matkit.core import mod_batch, mod_utility, mod_argparse, mod_incr_load

'''-----------------------------------------------------------------------------
                               MODULE VARIABLES
-----------------------------------------------------------------------------'''
app_name = 'app_incr_load.py'

'''-----------------------------------------------------------------------------
                                MAIN PROGRAM
-----------------------------------------------------------------------------'''
if __name__ == '__main__':

    ##########
    # Parser #
    ##########

    # Main argument parser
    parser_main = argparse.ArgumentParser(
        description='Perform incremental loading for theoretical/ideal strength' \
            ' calculations.')

    # Create sub parsers
    parser_mode = parser_main.add_subparsers(
        title='Run mode', dest='run_mode', help='Specify run mode. One of the'\
        ' following {SETUP_LOAD, SETUP_FORCE_CONSTANTS, RUN, PROCESS, PLOT_STRENGTH, EXTRACT_RESULTS}',
        required=True)

    # Create a parser for the SETUP_LOAD command and add arguments
    parser_SETUP_LOAD = parser_mode.add_parser('SETUP_LOAD')
    mod_argparse.add_arguments(parser_SETUP_LOAD, ['job_name', 'work_dir', \
        'arg_list', 'wall_time_sec', 'setup_filename', 'is_final_static', 
        'is_taskfarm', 'n_taskfarm', 'taskfarm_n_proc', 'partition', 'qos']) 
        
    # Create a parser for the SETUP_FORCE_CONSTANTS command and add arguments
    parser_SETUP_FORCE_CONSTANTS = parser_mode.add_parser('SETUP_FORCE_CONSTANTS')
    mod_argparse.add_arguments(parser_SETUP_FORCE_CONSTANTS, ['job_name', \
        'work_dir', 'wall_time_sec', 'is_taskfarm', 'n_taskfarm', \
        'taskfarm_n_proc', 'partition', 'qos'])  
        
    # Create a parser for the RUN command and add arguments
    # NOTE: It only runs in taskfarm mode for this app
    parser_RUN = parser_mode.add_parser('RUN')
    mod_argparse.add_arguments(parser_RUN, ['work_dir', 'processing_mode'])               

    # Create a parser for the PROCESS command and add arguments
    parser_PROCESS = parser_mode.add_parser('PROCESS')
    mod_argparse.add_arguments(parser_PROCESS, ['calculation_type', 'work_dir'])
    
    # Create a parser for the PROCESS command and add arguments
    parser_EXTRACT_RESULTS = parser_mode.add_parser('EXTRACT_RESULTS')
    mod_argparse.add_arguments(parser_EXTRACT_RESULTS, ['calculation_type', 'work_dir'])
    
    # Create a parser for the PLOT STRENGTH at a particular superposded or free stress case
    parser_PLOT_STRENGTH = parser_mode.add_parser('PLOT_STRENGTH')
    mod_argparse.add_arguments(parser_PLOT_STRENGTH, ['calculation_type', 'work_dir'])
    
    # Parse arguments and process the arguments
    args = parser_main.parse_args()
    mod_argparse.arg_process(args, app_name)

    #################
    # End of parser #
    #################

    #########################################
    # Setup incremental loading calculation #
    #########################################
    if args.run_mode == 'SETUP_LOAD':

        # Check if work_dir exists, if so warn and move to it.
        args.work_dir = args.work_dir + '/LOAD'
        mod_utility.warn_create_dir(args.work_dir)
        os.chdir(args.work_dir)

        # Import data setup file, other data will be directly used
        setup = importlib.import_module(args.setup_filename)
        element = args.arg_list[0]
        element_incr_load_info_list = setup.get_element_incr_load_info_list(element)  
    
        # Batch object
        batch_object = mod_batch.Scheduler_Info(
            job_name = args.job_name,
            mem_per_cpu_mb=CONFIG.MEM_PER_CPU_MB,
            partition=args.partition, qos=args.qos,
            time_seconds=args.wall_time_sec)
            
        # Setup DFT inputs
        incr_load_rel_dir_list = []
        for incr_load_info in element_incr_load_info_list:
            mod_incr_load.setup_incr_load(
                work_dir=args.work_dir,
                incr_load_info=incr_load_info,
                batch_object=batch_object)
            incr_load_rel_dir_list.append(incr_load_info['rel_dir_name'])
        with open('incr_load_rel_dir_info.json', 'w') as fh:
            json.dump({'incr_load_rel_dir_list' : incr_load_rel_dir_list}, fh, indent=4)
                
        # Get all waiting directories
        abs_job_dir_list = mod_utility.get_waiting_dirs(root_dir=args.work_dir)

        # Create taskfarms
        if args.is_taskfarm:
            mod_batch.create_taskfarm(
                batch_object=batch_object,
                n_taskfarm=args.n_taskfarm,
                taskfarm_n_proc=args.taskfarm_n_proc,
                abs_job_dir_list=abs_job_dir_list,
                work_dir=args.work_dir)
                
    #####################################
    # Setup force constant calculations #
    #####################################
    elif args.run_mode == 'SETUP_FORCE_CONSTANTS':
    
        abs_load_base_dir = args.work_dir + '/LOAD'
        abs_force_constants_base_dir = args.work_dir + '/FORCE_CONSTANTS'

        # The LOAD directory with already finished LOAD calculations must exist
        mod_utility.error_check_dir_exists(
            dirname=abs_load_base_dir, module=app_name,
            subroutine='SETUP_FORCE_CONSTANTS')

        # Batch object
        batch_object = mod_batch.Scheduler_Info(
            job_name = args.job_name,
            mem_per_cpu_mb=CONFIG.mem_per_cpu_mb,
            partition=args.partition, qos=args.qos,
            time_seconds=args.wall_time_sec)
            
        # Read the relative directories of each incremental loading job
        with open(abs_load_base_dir + '/incr_load_rel_dir_info.json', 'r') as fh:
            incr_load_rel_dir_list_dict = json.load(fh)

        # The force constants should have the same relative directory name as that of the incremental loading
        for rel_dir in incr_load_rel_dir_list_dict['incr_load_rel_dir_list']:
            mod_incr_load.setup_force_constant_calculations(
                abs_incr_load_dir=abs_load_base_dir + '/' + rel_dir,
                abs_force_constants_dir=abs_force_constants_base_dir + '/' + rel_dir)
                
        # Get all waiting directories
        abs_job_dir_list = mod_utility.get_waiting_dirs(root_dir=abs_force_constants_base_dir)

        # Create taskfarms
        if args.is_taskfarm:
            mod_batch.create_taskfarm(
                batch_object=batch_object,
                n_taskfarm=args.n_taskfarm,
                taskfarm_n_proc=args.taskfarm_n_proc,
                abs_job_dir_list=abs_job_dir_list,
                work_dir=abs_force_constants_base_dir)                

    #######################
    # Run the calculation #
    #######################
    elif args.run_mode == 'RUN':
     
       # NOTE: This app runs in taskfarm mode ONLY
       mod_batch.run_batch_jobs(root_dir=args.work_dir, run_mode=args.processing_mode)

    #######################
    # Process the results #
    #######################
    elif args.run_mode == 'PROCESS':

        os.chdir(args.work_dir)
        
        # Process incremental loading
        if args.calculation_type == 'LOAD':

            with open('incr_load_rel_dir_info.json', 'r') as fh:
                incr_load_rel_dir_list_dict = json.load(fh)

            for rel_dir in incr_load_rel_dir_list_dict['incr_load_rel_dir_list']:
                mod_incr_load.process_incr_load(
                    work_dir=rel_dir)
        
        # Process force constants
        elif args.calculation_type == 'FORCE_CONSTANTS':
            print('not yet implemented in the open-sourced version')

    ############################
    # extract multiple results #
    ############################
    elif args.run_mode == 'EXTRACT_RESULTS':

        os.chdir(args.work_dir)
       
        # Process incremental loadings
        if args.calculation_type == 'LOAD':

            with open('incr_load_rel_dir_info.json', 'r') as fh:
                incr_load_rel_dir_list_dict = json.load(fh)
                
            # Extract incremental loading families
            incr_load_all_results = {
            }

            # Each relative directory has the following structure family/STRESS<I><J><K>
            for rel_dir in incr_load_rel_dir_list_dict['incr_load_rel_dir_list']:
                rel_dir_split = os.path.split(rel_dir)
                
                # If the stress directory key does not exist, create it
                if rel_dir_split[0] not in incr_load_all_results:
                    incr_load_all_results[rel_dir_split[0]] = []
                    
                result_dict = {
                    rel_dir_split[1] : mod_incr_load.read_results(rel_dir)
                }
                incr_load_all_results[rel_dir_split[0]].append(result_dict)
                
            with open('incr_load_all_results.json', 'w') as fh:
                json.dump(incr_load_all_results, fh, indent=4,
                    cls=mod_utility.json_numpy_encoder)
            
    ####################
    # Plot the results #
    ####################
    elif args.run_mode == 'PLOT_STRENGTH':
    
        '''
          Given the ideal strength calculation processed results of various
          loading systems of a lattice type at a particular superposed stress
          value, this subroutine plots the stress vs strain and energy vs starin
          plots.
        '''
       
        # Process incremental loading
        if args.calculation_type == 'LOAD':

            mod_incr_load.plot_loading(work_dir=args.work_dir)
        
        # Process force constants
        elif args.calculation_type == 'PHONON':
            print('not yet implemented')   
                
    ###########
    # DEFAULT #
    ###########
    else:

        sys.stderr.write("Error: In calculation %s\n" %(app_name))
        sys.stderr.write("       Unknown run_mode %s\n" %(args.run_mode))
        sys.stderr.write("       Allowed run_mode {SETUP, RUN, PROCESS}\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

'''----------------------------------------------------------------------------
                               END OF PROGRAM
----------------------------------------------------------------------------'''
