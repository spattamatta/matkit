'''----------------------------------------------------------------------------
                           app_as_is_automator.py

 Description: Perform basic dft of a structure or a set of structures as per
   the inputs and without any specialized workflows.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import argparse
import importlib

# Externally installed modules
# None

# Local imports
from matkit import CONFIG
from matkit.core import mod_batch, mod_utility, mod_argparse
from matkit.calculators import mod_dft

'''----------------------------------------------------------------------------
                                MODULE VATIABLES
----------------------------------------------------------------------------'''
app_name = 'app_as_it_is_automator.py'

'''----------------------------------------------------------------------------
                                SUBROUTINES
----------------------------------------------------------------------------'''
# None

'''----------------------------------------------------------------------------
                                MAIN PROGRAM
----------------------------------------------------------------------------'''
if __name__ == '__main__':

    ##########
    # Parser #
    ##########

    # Main argument parser
    parser_main = argparse.ArgumentParser(
        description='Perform as it is calculations.')

    # Create sub parsers
    parser_mode = parser_main.add_subparsers(
        title='Run mode', dest='run_mode', help='Specify run mode.',
        required=True)

    # Create a parser for the SETUP command and add arguments
    parser_SETUP = parser_mode.add_parser("SETUP")
    mod_argparse.add_arguments(parser_SETUP, ['work_dir', 'setup_filename',
        'wall_time_sec', 'is_taskfarm', 'n_taskfarm', 'taskfarm_n_proc',
        'parset_filename', 'qos',  'partition'])

    # Create a parser for the RUN command and add arguments
    parser_RUN = parser_mode.add_parser('RUN')
    mod_argparse.add_arguments(parser_RUN, ['work_dir', 'processing_mode'])

    # Create a parser for the PROCESS command and add arguments
    parser_PROCESS = parser_mode.add_parser('PROCESS')
    mod_argparse.add_arguments(parser_PROCESS, ['work_dir'])
    
    # Create a parser for the TREE CLEAN
    parser_TREE_CLEAN = parser_mode.add_parser('TREE_CLEAN')
    mod_argparse.add_arguments(parser_TREE_CLEAN, ['work_dir', 'clean_level'])    

    # Parse arguments and process the arguments
    args = parser_main.parse_args()
    mod_argparse.arg_process(args, app_name)

    #################
    # End of parser #
    #################

    #########
    # Setup #
    #########
    if args.run_mode == 'SETUP':

        # Check if work_dir exists, if so warn and move to it.
        mod_utility.warn_create_dir(args.work_dir)
        os.chdir(args.work_dir)

        # Import data setup file, other data will be directly used
        setup = importlib.import_module(args.setup_filename)
                
        # Batch object
        batch_object = mod_batch.Scheduler_Info(
            mem_per_cpu_mb=CONFIG.MEM_PER_CPU_MB, partition=args.partition,
            qos=args.qos, time_seconds=args.wall_time_sec)

        # Set up DFT inputs
        mod_dft.setup_basic_dft_calculations(
            work_dir = args.work_dir,
            rel_dir_list = setup.rel_dir_list,
            structure_list = setup.structure_list,
            calculator_input_list = setup.calculator_input_list,
            sim_info_list = setup.sim_info_list,
            batch_object = batch_object,
            is_taskfarm=args.is_taskfarm,
            n_taskfarm=args.n_taskfarm,
            taskfarm_n_proc=args.taskfarm_n_proc)

    #######################
    # Run the calculation #
    #######################
    elif args.run_mode == 'RUN':
    
        # This app runs only in batch mode
        mod_batch.run_batch_jobs(root_dir=args.work_dir, run_mode=args.processing_mode)

    #######################
    # Process the results #
    #######################
    elif args.run_mode == 'PROCESS':

        mod_dft.process_basic_dft_calculations(work_dir=args.work_dir)
        
    ##############
    # TREE CLEAN #
    ##############
    elif args.run_mode == "TREE_CLEAN":

        mod_calculator.tree_clean(work_dir=args.work_dir,
                                  clean_level=args.clean_level)        

    ###########
    # DEFAULT #
    ###########
    else:

        sys.stderr.write("Error: In calculation %s\n" %(app_name))
        sys.stderr.write("       Unknown run_mode %s\n" %(args.run_mode))
        sys.stderr.write("       Allowed run_mode {SETUP, RUN, POSTPROCESS}\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

'''----------------------------------------------------------------------------
                               END OF PROGRAM
----------------------------------------------------------------------------'''
