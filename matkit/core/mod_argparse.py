'''----------------------------------------------------------------------------
                               mod_argparse.py

 Description: Handy module for frequently used argument parsers.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import sys
import argparse

# Externally installed modules
# None

# Local imports
from matkit import CONFIG
from matkit.core import mod_utility

'''----------------------------------------------------------------------------
                                MODULE VATIABLES
----------------------------------------------------------------------------'''
module_name = "mod_argparse.py"

'''----------------------------------------------------------------------------
                                SUBROUTINES
----------------------------------------------------------------------------'''

##################
### SUBROUTINE ###
##################

def arg_process(args, caller_name):

    '''
      If task farming option is set, then the total number of processors for
      the taskfarm are required.
      This check is only performed in the seup mode
    '''
    
    # Add base workdir to workdir
    args.work_dir = CONFIG.WORK_DIR_BASE + "/" + args.work_dir
    
    if args.run_mode == "SETUP":
    
        if hasattr(args, 'is_taskfarm'):
            # If is_taskfamr is true then # or processors for taskfarm need to be specified
            if args.is_taskfarm:
                if args.taskfarm_n_proc == -1:
                    sys.stderr.write("Error: In calculation '%s'\n" %(caller_name))
                    sys.stderr.write("       If in SETUP, is_taskfarm=True then " \
                        "taskfarm_n_proc must be specified\n")
                    sys.stderr.write("       Terminating!!!\n")
                    exit(1)
                
    # Partition and qos for batch processing
    if hasattr(args, 'partition'):
        if args.partition is None:
            args.partition = CONFIG.PARTITION

    if hasattr(args, 'qos'):
        if args.qos is None:
            args.qos = CONFIG.QOS   


##################
### SUBROUTINE ###
##################

def add_arguments(parser, arg_list):

    for arg in arg_list:
    
        if arg == "job_name":
            add_arg_job_name(parser)
            
        elif arg == "partition":
            add_arg_partition(parser)
            
        elif arg == "qos":
            add_arg_qos(parser)                      

        elif arg == "n_proc":
            add_arg_n_proc(parser)
            
        elif arg == "n_proc_per_atom":
            add_arg_n_proc_per_atom(parser)            

        elif arg == "n_relax_iter":
            add_arg_n_relax_iter(parser)

        elif arg == "relax_type":
            add_arg_relax_type(parser)

        elif arg == "work_dir":
            add_arg_work_dir(parser)

        elif arg == "plot_dir":
            add_arg_plot_dir(parser)

        elif arg == "mem_per_cpu_mb":
            add_arg_mem_per_cpu_mb(parser)

        elif arg == "wall_time_sec":
            add_arg_wall_time_sec(parser)

        elif arg == "material":
            add_arg_material(parser)

        elif arg == "is_multi_material":
            add_arg_is_multi_material(parser)

        elif arg == "n_core":
            add_arg_n_core(parser)

        elif arg == "is_initial_relax":
            add_arg_is_initial_relax(parser)

        elif arg == "is_final_static":
            add_arg_is_final_static(parser)

        elif arg == "is_taskfarm":
            add_arg_is_taskfarm(parser)
            
        elif arg == "n_taskfarm":
            add_arg_n_taskfarm(parser)

        elif arg == "taskfarm_n_proc":
            add_arg_taskfarm_n_proc(parser)

        elif arg == "is_merge_taskfarm":
            add_arg_is_merge_taskfarm(parser)
            
        elif arg == "n_merged_taskfarm":
            add_arg_n_merged_taskfarm(parser)

        elif arg == "merged_taskfarm_n_proc":
            add_arg_merged_taskfarm_n_proc(parser)

        elif arg == "processing_mode":
            add_arg_processing_mode(parser)

        elif arg == "parset_filename":
            add_arg_parset_filename(parser)

        elif arg == "setup_filename":
            add_arg_setup_filename(parser)

        elif arg == "input_filename":
            add_arg_input_filename(parser)

        elif arg == "input_prefix":
            add_arg_input_prefix(parser)

        elif arg == "output_filename":
            add_arg_output_filename(parser)

        elif arg == "output_prefix":
            add_arg_output_prefix(parser)

        elif arg == "plot_filename":
            add_arg_plot_filename(parser)

        elif arg == "elasticity_order":
            add_arg_elasticity_order(parser)

        elif arg == "EOS_filename":
            add_arg_EOS_filename(parser)

        elif arg == "EOS_filename_list":
            add_arg_EOS_filename_list(parser)
            
        elif arg == "deformation_parameter":
            add_arg_deformation_parameter(parser)            

        elif arg == "EC_filename":
            add_arg_EC_filename(parser)

        elif arg == "EC_filename_list":
            add_arg_EC_filename_list(parser)

        elif arg == "SOEC_filename":
            add_arg_SOEC_filename(parser)

        elif arg == "SOEC_filename_list":
            add_arg_SOEC_filename_list(parser)

        elif arg == "TOEC_filename":
            add_arg_TOEC_filename(parser)

        elif arg == "TOEC_filename_list":
            add_arg_TOEC_filename_list(parser)

        elif arg == "orient_filename":
            add_arg_orient_filename(parser)

        elif arg == "loading_type":
            add_arg_loading_type(parser)

        elif arg == "temperature_list":
            add_arg_temperature_list(parser)

        elif arg == "pressure_list":
            add_arg_pressure_list(parser)

        elif arg == "pressure_range":
            add_arg_pressure_range(parser)

        elif arg == "pressure":
            add_arg_pressure(parser)

        elif arg == "hydure_list":
            add_arg_hydure_list(parser)

        elif arg == "hydure":
            add_arg_hydure(parser)

        elif arg == "n_strain_points":
            add_arg_n_strain_points(parser)

        elif arg == "compressive_n_strain_points":
            add_arg_compressive_n_strain_points(parser)

        elif arg == "expansive_n_strain_points":
            add_arg_expansive_n_strain_points(parser)

        elif arg == "strain_spacing_type":
            add_arg_strain_spacing_type(parser)

        elif arg == "strain_spacing_coefficient":
            add_arg_strain_spacing_coefficient(parser)

        elif arg == "volumetric_strain_percent":
            add_arg_volumetric_strain_percent(parser)

        elif arg == "compressive_volumetric_strain_percent":
            add_arg_compressive_volumetric_strain_percent(parser)

        elif arg == "expansive_volumetric_strain_percent":
            add_arg_expansive_volumetric_strain_percent(parser)

        elif arg == "fit_order":
            add_arg_fit_order(parser)

        elif arg == "n_thread":
            add_arg_n_thread(parser)

        elif arg == "comment_list":
            add_arg_comment_list(parser)

        elif arg == "arg_list":
            add_arg_arg_list(parser)

        elif arg == "clean_level":
            add_arg_clean_level(parser)

        elif arg == "calculation_type":
            add_arg_calculation_type(parser)
            
        elif arg == "method":
            add_arg_method(parser)
            
        elif arg == "input_type":
            add_arg_input_type(parser)

        else:
            sys.stderr.write("Error: In module '%s'\n" %(module_name))
            sys.stderr.write("       In subroutine 'add_arguments'\n")
            sys.stderr.write("       Unknown argument type '%s' in input argument arg_list\n" %(arg))
            sys.stderr.write("       Terminating!!!\n")
            exit(1)
            
##################
### SUBROUTINE ###
##################

def create_parser_job_name( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_job_name(parser)
    return parser

def add_arg_job_name(parser):

    parser.add_argument(
        "--job_name", "-jname", type=str, help="Enter the name of the job.\
        Default value is 'matpad'", default='matpad', action="store",
        dest='job_name')
        
##################
### SUBROUTINE ###
##################

def create_parser_partition( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_partition(parser)
    return parser

def add_arg_partition(parser):

    parser.add_argument(
        '--partition', '-partition', type=str, help='Enter the name of the ' \
        'slurm partition. Default value is None', default=None, action='store',
        dest='partition')
        
##################
### SUBROUTINE ###
##################

def create_parser_qos( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_qos(parser)
    return parser

def add_arg_qos(parser):

    parser.add_argument(
        '--qos', '-qos', type=str, help='Enter the quality of service of ' \
        'slurm job. Default value is None', default=None, action='store',
        dest='qos')                        

##################
### SUBROUTINE ###
##################

def create_parser_n_proc( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_proc(parser)
    return parser

def add_arg_n_proc(parser):

    parser.add_argument(
        "--n_proc", "-np", type=int, help="Enter the number of processors for \
        each relaxation calculation. Default value is 1.", default=1,
        action="store", dest='n_proc')
        
##################
### SUBROUTINE ###
##################

def create_parser_n_proc_per_atom( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_proc_per_atom(parser)
    return parser

def add_arg_n_proc_per_atom(parser):

    parser.add_argument(
        '--n_proc_per_atom', '-nppatom', type=int, help='Enter the number of' \
        ' processors per atom. Default value is 1.', default=1, action="store",
        dest='n_proc_per_atom')        

##################
### SUBROUTINE ###
##################

def create_parser_n_relax_iter( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_relax_iter(parser)
    return parser

def add_arg_n_relax_iter(parser):

    parser.add_argument(
        "--n_relax_iter", "-nr", type=int, help="Enter the number of \
        relaxation iterations. Default value is 2.",
        default=2, action="store", dest='n_relax_iter')

##################
### SUBROUTINE ###
##################

def create_parser_relax_type( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_relax_type(parser)
    return parser

def add_arg_relax_type(parser):

    parser.add_argument(
        "--relax_type", "-relaxtype", type=str, help="Enter the type of \
        relaxation. Allowed types: 'relax_isv', 'relax_i', 'relax_iv'",
        required=True, action="store", dest='relax_type')

##################
### SUBROUTINE ###
##################

def create_parser_work_dir( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_work_dir(parser)
    return parser

def add_arg_work_dir(parser):

    parser.add_argument(
        "--work_dir", "-wdir", type=str, help="Enter the location of the work \
        directory. Reports warning if it already exists. The user can then \
        choose to overwrite.", required=True, action="store", dest='work_dir')

##################
### SUBROUTINE ###
##################

def create_parser_plot_dir( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_plot_dir(parser)
    return parser

def add_arg_plot_dir(parser):

    parser.add_argument(
        '--plot_dir', '-plotdir', type=str, help='Enter the location of the '\
        'plot directory to store plots.', required=True, action="store",
        dest='plot_dir')
        
##################
### SUBROUTINE ###
##################


def create_parser_mem_per_cpu_mb( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_work_dir(parser)
    return parser

def add_arg_mem_per_cpu_mb(parser):

    parser.add_argument(
        "--mem_per_cpu_mb", "-mem_mb", type=int, help="Enter the memory per \
        cpu in mega bytes.", default=4000, action="store",
        dest='mem_per_cpu_mb')

##################
### SUBROUTINE ###
##################


def create_parser_wall_time_sec( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_wall_time_sec(parser)
    return parser

def add_arg_wall_time_sec(parser):

    parser.add_argument(
        "--wall_time_sec", "-wall_sec", type=int, help="Enter the wall time \
        to be requested in seconds.", default=3600, action="store",
        dest='wall_time_sec')

##################
### SUBROUTINE ###
##################


def create_parser_material( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_material(parser)
    return parser

def add_arg_material(parser):

    parser.add_argument(
        "--material", "-mat", type=str, help="Enter material file location. \
        The material file should contain an ASE Atoms object called 'atoms' \
        and an ASE calculator object called 'calc'. The calculator should be \
        of same type of the calculator variable in CONFIG.py i.e. either \
        'vasp' or 'qe'. NOTE: If operating under multiple material mode, the \
        material file needs to supply an array of atoms called atoms_arr and \
        an array of calculators called calc_arr", required=True,
        action="store", dest='material')

##################
### SUBROUTINE ###
##################


def create_parser_is_multi_material( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_is_multi_material(parser)
    return parser

def add_arg_is_multi_material(parser):

    parser.add_argument(
        "--is_multi_material", "-ismmat", type=mod_utility.str2bool,
        help="If the material file is a multiple material file, which \
        supplies arrays of atoms and corresponding calculator objects",
        default=False, action="store", dest='is_multi_material')

##################
### SUBROUTINE ###
##################


def create_parser_n_core( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_core(parser)
    return parser

def add_arg_n_core(parser):

    parser.add_argument(
        "--n_core", "-nc", type=int, help="Enter the number of cores for each \
        calculation. This option is specific to vasp. The recommended setting \
        is n_core = sqrt(n_proc)", default=1, action="store", dest='n_core')

##################
### SUBROUTINE ###
##################


def create_parser_is_initial_relax( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_is_initial_relax(parser)
    return parser


def add_arg_is_initial_relax(parser):

    parser.add_argument(
        "--is_initial_relax", "-iir", type=mod_utility.str2bool,
        help="Specify if initial relaxation is needed.", default=False,
        action='store', dest='is_initial_relax')

##################
### SUBROUTINE ###
##################


def create_parser_is_final_static( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_is_final_static(parser)
    return parser


def add_arg_is_final_static(parser):

    parser.add_argument(
        "--is_final_static", "-ifs", type=mod_utility.str2bool,
        help="Specify if a final static calculation is needed after each \
        relaxation.", default=True, action='store', dest='is_final_static')

##################
### SUBROUTINE ###
##################


def create_parser_is_taskfarm( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_is_taskfarm(parser)
    return parser

def add_arg_is_taskfarm(parser):

    parser.add_argument(
        "--is_taskfarm", "-tf", type=mod_utility.str2bool,
        help="Specify if running in a task farming mode. Default value is \
        False.", default=False, action="store", dest='is_taskfarm')
        
##################
### SUBROUTINE ###
##################


def create_parser_n_taskfarm( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_taskfarm(parser)
    return parser

def add_arg_n_taskfarm(parser):

    parser.add_argument(
        "--n_taskfarm", "-ntf", type=int, help="Enter the number of \
        taskfarms. Default value is 1.", default=1, action="store",
        dest='n_taskfarm')

##################
### SUBROUTINE ###
##################


def create_parser_taskfarm_n_proc( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_taskfarm_n_proc(parser)
    return parser

def add_arg_taskfarm_n_proc(parser):

    parser.add_argument(
        "--taskfarm_n_proc", "-tfnp", type=int, help="Enter the number of \
        processors for the entire task farm. Must be specified if --taskfarm \
        =True. Default value is -1.", default=-1, action="store",
        dest='taskfarm_n_proc')

##################
### SUBROUTINE ###
##################


def create_parser_is_merge_taskfarm( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_is_merge_taskfarm(parser)
    return parser

def add_arg_is_merge_taskfarm(parser):

    parser.add_argument(
        "--is_merge_taskfarm", "-mtf", type=mod_utility.str2bool,
        help="Specify if running in a merge task farming mode. Default value is \
        False.", default=False, action="store", dest='is_merge_taskfarm')
        
##################
### SUBROUTINE ###
##################


def create_parser_n_merged_taskfarm( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_merged_taskfarm(parser)
    return parser

def add_arg_n_merged_taskfarm(parser):

    parser.add_argument(
        "--n_merged_taskfarm", "-nmtf", type=int, help="Enter the number of \
        merged taskfarms. Default value is 1.", default=1, action="store",
        dest='n_merged_taskfarm')

##################
### SUBROUTINE ###
##################


def create_parser_merged_taskfarm_n_proc( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_merged_taskfarm_n_proc(parser)
    return parser

def add_arg_merged_taskfarm_n_proc(parser):

    parser.add_argument(
        "--merged_taskfarm_n_proc", "-mtfnp", type=int, help="Enter the " \
        "number of processors for the entire task farm. Must be specified " \
        "if --merged_taskfarm =True. Default value is -1.", default=-1,
        action="store", dest='merged_taskfarm_n_proc')

##################
### SUBROUTINE ###
##################


def create_parser_processing_mode( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_processing_mode(parser)
    return parser

def add_arg_processing_mode(parser):

    parser.add_argument(
        "--processing_mode", "-pm", type=str,
        help="Specify if stand_alone, batch or taskfarm mode of processing.",
        required=True, action='store', dest='processing_mode')

##################
### SUBROUTINE ###
##################


def create_parser_parset_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_parset_filename(parser)
    return parser

def add_arg_parset_filename(parser):

    parser.add_argument(
        "--parset_filename", "-parset", type=str, help="Enter the file of the \
        parallel settings file", default=None, action="store",
        dest='parset_filename')

##################
### SUBROUTINE ###
##################


def create_parser_setup_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_setup_filename(parser)
    return parser

def add_arg_setup_filename(parser):

    parser.add_argument(
        "--setup_filename", "-setup", type=str, help="Enter the setup \
        filename",  default='', action="store", dest='setup_filename')

##################
### SUBROUTINE ###
##################


def create_parser_input_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_input_filename(parser)
    return parser

def add_arg_input_filename(parser):

    parser.add_argument(
        "--input_filename", "-inflnm", type=str, help="Enter the input \
        filename",  required=True, action="store", dest='input_filename')

##################
### SUBROUTINE ###
##################


def create_parser_input_prefix( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_input_prefix(parser)
    return parser

def add_arg_input_prefix(parser):

    parser.add_argument(
        "--input_prefix", "-inprefix", type=str, help="Enter the input \
        filename",  required=True, action="store", dest='input_prefix')

##################
### SUBROUTINE ###
##################


def create_parser_output_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_output_filename(parser)
    return parser

def add_arg_output_filename(parser):

    parser.add_argument(
        "--output_filename", "-outflnm", type=str, help="Enter the output \
        filename",  required=True, action="store", dest='output_filename')

##################
### SUBROUTINE ###
##################


def create_parser_output_prefix( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_output_prefix(parser)
    return parser

def add_arg_output_prefix(parser):

    parser.add_argument(
        "--output_prefix", "-outprefix", type=str, help="Enter the output \
        filename",  required=True, action="store", dest='output_prefix')

##################
### SUBROUTINE ###
##################


def create_parser_plot_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_plot_filename(parser)
    return parser

def add_arg_plot_filename(parser):

    parser.add_argument(
        "--plot_filename", "-plotflnm", type=str, help="Enter the plot \
        filename",  required=True, action="store", dest='plot_filename')

##################
### SUBROUTINE ###
##################

def create_parser_elasticity_order( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_elasticity_order(parser)
    return parser

def add_arg_elasticity_order(parser):

    parser.add_argument(
        "--elasticity_order", "-ecorder", type=int, help="Enter the order of \
        elastic constants required. Currently 2 or 3.", required=True,
        action="store", dest='elasticity_order')

##################
### SUBROUTINE ###
##################


def create_parser_EOS_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_EOS_filename(parser)
    return parser

def add_arg_EOS_filename(parser):

    parser.add_argument(
        "--EOS_filename", "-eos", type=str, help="Enter the equation of \
        state (EOS) filename",  default=None, action="store",
        dest='EOS_filename')

##################
### SUBROUTINE ###
##################

def create_parser_EOS_filename_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_EOS_filename_list(parser)
    return parser

def add_arg_EOS_filename_list(parser):

    parser.add_argument(
        "--EOS_filename_list", "-eosfilelist", type=str, help='Enter the equation of \
        state (EOS) filename list', default=None, nargs='+', action="store",
        dest='EOS_filename_list')
        
##################
### SUBROUTINE ###
##################


def create_parser_deformation_parameter( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_deformation_parameter(parser)
    return parser

def add_arg_deformation_parameter(parser):

    parser.add_argument(
        "--deformation_parameter", "-defpar", type=float, help="Enter the deformation \
        parameter", required=True, action="store",
        dest="deformation_parameter")

##################
### SUBROUTINE ###
##################


def create_parser_EC_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_EC_filename(parser)
    return parser

def add_arg_EC_filename(parser):

    parser.add_argument(
        "--EC_filename", "-ec", type=str, help="Enter the elastic constants \
        filename",  required=True, action="store", dest='EC_filename')

##################
### SUBROUTINE ###
##################


def create_parser_EC_filename_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_EC_filename_list(parser)
    return parser

def add_arg_EC_filename_list(parser):

    parser.add_argument(
        "--EC_filename_list", "-ecfilelist", type=str, help="Enter the \
        elastic constants filename list",  required=True, nargs='+1',
        action="store", dest='EC_filename_list')

##################
### SUBROUTINE ###
##################


def create_parser_SOEC_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_SOEC_filename(parser)
    return parser

def add_arg_SOEC_filename(parser):

    parser.add_argument(
        "--SOEC_filename", "-soec", type=str, help="Enter the second order \
        elastic constants (SOEC) filename",  required=True, action="store",
        dest='SOEC_filename')

##################
### SUBROUTINE ###
##################


def create_parser_SOEC_filename_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_SOEC_filename_list(parser)
    return parser

def add_arg_SOEC_filename_list(parser):

    parser.add_argument(
        "--SOEC_filename_list", "-soecfilelist", type=str, help="Enter the second \
        order elastic constants (SOEC) filename list",  default=None,
        action="store", nargs='+', dest='SOEC_filename_list')

##################
### SUBROUTINE ###
##################


def create_parser_TOEC_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_TOEC_filename(parser)
    return parser

def add_arg_TOEC_filename(parser):

    parser.add_argument(
        "--TOEC_filename", "-toec", type=str, help="Enter the third order \
        elastic constants (TOEC) filename",  required=True, action="store",
        dest='TOEC_filename')

##################
### SUBROUTINE ###
##################


def create_parser_TOEC_filename_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_TOEC_filename_list(parser)
    return parser

def add_arg_TOEC_filename_list(parser):

    parser.add_argument(
        "--TOEC_filename_list", "-toeclist", type=str, help="Enter the third \
        order elastic constants (TOEC) filename list",  required=True,
        action="store", nargs='+', dest='TOEC_filename_list')

##################
### SUBROUTINE ###
##################


def create_parser_orient_filename( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_orient_filename(parser)
    return parser

def add_arg_orient_filename(parser):

    parser.add_argument(
        "--orient_filename", "-oriflnm", type=str, help="Enter the grain \
        orientations filename",  default=None, action="store",
        dest='orient_filename')

##################
### SUBROUTINE ###
##################

def create_parser_loading_type( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_loading_type(parser)
    return parser

def add_arg_loading_type(parser):

    parser.add_argument(
        "--loading_type", "-loadtype", type=str, help="Enter the type of \
        loading. Allowed types: 'STRESS', 'STRAIN'", required=True,
        action="store", dest='loading_type')

##################
### SUBROUTINE ###
##################


def create_parser_temperature_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_temperature_list(parser)
    return parser

def add_arg_temperature_list(parser):

    parser.add_argument(
        "--temperature_list", "-Tlist", type=float, help="Enter the temperature \
        of the simulation in Kelvin. To enter multiple temperatures ex: \
        --temperature_list 0.0 100.0 200.0 300.0", default=[], nargs="+",
        dest="temperature_list")

##################
### SUBROUTINE ###
##################


def create_parser_pressure_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_pressure_list(parser)
    return parser

def add_arg_pressure_list(parser):

    parser.add_argument(
        "--pressure_list", "-Plist", type=float, help="Enter the pressure \
        of the simulation in GPa. To enter multiple pressures ex: \
        --pressure_list 0.0 10.0 20.0 30.0", default=[], nargs="+",
        dest="pressure_list")

##################
### SUBROUTINE ###
##################


def create_parser_pressure_range( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_pressure_range(parser)
    return parser

def add_arg_pressure_range(parser):

    parser.add_argument(
        "--pressure_range", "-Prange", type=float, help="Enter the pressure \
        range the simulation in GPa. Ex: --pressure_arnge -10.0 10.0",
        default=[], nargs="+", dest="pressure_range")

##################
### SUBROUTINE ###
##################


def create_parser_pressure( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_pressure(parser)
    return parser

def add_arg_pressure(parser):

    parser.add_argument(
        "--pressure", "-pressure", type=float, help="Enter the pressure \
        of the simulation in GPa", required=True, action="store",
        dest="pressure")

##################
### SUBROUTINE ###
##################


def create_parser_hydure_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_hydure_list(parser)
    return parser

def add_arg_hydure_list(parser):

    parser.add_argument(
        "--hydure_list", "-hydlist", type=float, help="Enter the hydure \
        of the simulation in GPa. To enter multiple hydures ex: \
        --hydure_list 0.0 10.0 20.0 30.0", default=[], nargs="+",
        dest="hydure_list")

##################
### SUBROUTINE ###
##################


def create_parser_hydure( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_hydure(parser)
    return parser

def add_arg_hydure(parser):

    parser.add_argument(
        "--hydure", "-hydure", type=float, help="Enter the hydure \
        of the simulation in GPa", required=True, action="store",
        dest="hydure")

##################
### SUBROUTINE ###
##################


def create_parser_n_strain_points( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_strain_points(parser)
    return parser

def add_arg_n_strain_points(parser):

    parser.add_argument(
        "--n_strain_points", "-nsp", type=int, help="Enter the number of \
        strain points. Default value is 15.", default=15, action="store",
        dest='n_strain_points')

##################
### SUBROUTINE ###
##################


def create_parser_compressive_n_strain_points( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_compressive_n_strain_points(parser)
    return parser

def add_arg_compressive_n_strain_points(parser):

    parser.add_argument(
        "--compressive_n_strain_points", "-lnsp", type=int, help="Enter the number of \
        strain points. Default value is 15.", default=15, action="store",
        dest='compressive_n_strain_points')

##################
### SUBROUTINE ###
##################


def create_parser_expansive_n_strain_points( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_expansive_n_strain_points(parser)
    return parser

def add_arg_expansive_n_strain_points(parser):

    parser.add_argument(
        "--expansive_n_strain_points", "-rnsp", type=int, help="Enter the number of \
        strain points. Default value is 15.", default=15, action="store",
        dest='expansive_n_strain_points')

##################
### SUBROUTINE ###
##################


def create_parser_strain_spacing_type( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_strain_spacing_type(parser)
    return parser

def add_arg_strain_spacing_type(parser):

    parser.add_argument(
        "--strain_spacing_type", "-sspt", type=str, help="Enter the type of \
        spacing of strain points. Default value is 'linear'.",
        default='linear', action="store", dest='strain_spacing_type')

##################
### SUBROUTINE ###
##################


def create_parser_strain_spacing_coefficient( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_strain_spacing_coefficient(parser)
    return parser

def add_arg_strain_spacing_coefficient(parser):

    parser.add_argument(
        "--strain_spacing_coefficient", "-sspc", type=float, help="Enter the \
        strain spacing coefficient. Default value is 1.0.",
        default=1.0, action="store", dest='strain_spacing_coefficient')

##################
### SUBROUTINE ###
##################


def create_parser_volumetric_strain_percent( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_volumetric_strain_percent(parser)
    return parser

def add_arg_volumetric_strain_percent(parser):

    parser.add_argument(
        "--volumetric_strain_percent", "-vsp", type=float, help="Enter the \
        volumetric strain percentage. Default value is 16.0. This corresponds \
        to a linear strain of 0.050717574498580165.", default=16.0,
        action="store", dest='volumetric_strain_percent')

##################
### SUBROUTINE ###
##################


def create_parser_compressive_volumetric_strain_percent( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_compressive_volumetric_strain_percent(parser)
    return parser

def add_arg_compressive_volumetric_strain_percent(parser):

    parser.add_argument(
        "--compressive_volumetric_strain_percent", "-lvsp", type=float, help="Enter the \
        volumetric strain percentage. Default value None.", default=None,
        action="store", dest='compressive_volumetric_strain_percent')

##################
### SUBROUTINE ###
##################


def create_parser_expansive_volumetric_strain_percent( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_expansive_volumetric_strain_percent(parser)
    return parser

def add_arg_expansive_volumetric_strain_percent(parser):

    parser.add_argument(
        "--expansive_volumetric_strain_percent", "-rvsp", type=float, help="Enter the \
        volumetric strain percentage. Default value is None.", default=None,
        action="store", dest='expansive_volumetric_strain_percent')

##################
### SUBROUTINE ###
##################


def create_parser_fit_order( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_fit_order(parser)
    return parser

def add_arg_fit_order(parser):

    parser.add_argument(
        "--fit_order", "-forder", type=int, help="Enter the order of curve \
        fitting.", required=True, action="store", dest='fit_order')

##################
### SUBROUTINE ###
##################


def create_parser_n_thread( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_n_thread(parser)
    return parser

def add_arg_n_thread(parser):

    parser.add_argument(
        "--n_thread", "-nt", type=int, help="Enter the number of threads. \
        The recommended setting  is n_thread=Number of cores on processors",
        default=1, action="store", dest='n_thread')

##################
### SUBROUTINE ###
##################

def create_parser_comment_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_comment_list(parser)
    return parser

def add_arg_comment_list(parser):

    parser.add_argument(
        "--comment_list", "-cmtlist", type=str, help='Enter some string ' \
        'comments.', default=None, nargs='+', action="store",
        dest='comment_list')

##################
### SUBROUTINE ###
##################

def create_parser_arg_list( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_arg_list(parser)
    return parser

def add_arg_arg_list(parser):

    parser.add_argument(
        "--arg_list", "-arglist", type=str, help='Enter some string ' \
        'arguments. The calling routine will have to cast the arguments ' \
        'into appropriate datatypes.', default=None, nargs='+', action="store",
        dest='arg_list')

##################
### SUBROUTINE ###
##################

def create_parser_clean_level( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_clean_level(parser)
    return parser

def add_arg_clean_level(parser):

    parser.add_argument(
        "--clean_level", "-clnlevel", type=str, help="Enter the level of \
        cleaning required. Allowed arguments: 'basic', 'dist', 'saver'.",
        required=True, action="store", dest='clean_level')
        
##################
### SUBROUTINE ###
##################

def create_parser_calculation_type( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_calculation_type(parser)
    return parser

def add_arg_calculation_type(parser):

    parser.add_argument(
        "--calculation_type", "-calctype", type=str, help="Enter the type of \
        calculation. Allowed types: Depends on the APP",
        required=True, action="store", dest='calculation_type')
        
##################
### SUBROUTINE ###
##################

def create_parser_method( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_method(parser)
    return parser

def add_arg_method(parser):

    parser.add_argument(
        "--method", "-method", type=str, help="Enter the method of \
        calculation. Allowed types: Depends on the APP", required=True,
        action="store", dest='method')
        
##################
### SUBROUTINE ###
##################

def create_parser_input_type( ):

    parser = argparse.ArgumentParser(add_help=False)
    add_arg_input_type(parser)
    return parser

def add_arg_method(parser):

    parser.add_argument(
        "--input_type", "-input_type", type=str, help="Enter the input \
        type. Allowed types: Depends on the APP", required=True,
        action="store", dest='input_type')        

'''----------------------------------------------------------------------------
                                 END OF MODULE
----------------------------------------------------------------------------'''
