'''-----------------------------------------------------------------------------
                                   CONFIG.py

 Description: Basic setup for work directory and calculator executable paths.

-----------------------------------------------------------------------------'''
import os
import sys
import socket
import importlib

__author__ = "A. S. L. Subrahmanyam Pattamatta"
__copyright__ = "Copyright 2024, MaTKit"
__version__ = "1.0"
__maintainer__ = "A. S. L. Subrahmanyam Pattamatta"
__email__ = "lalithasubrahmanyam@gmail.com"
__status__ = "Alpha"
__date__ = "Sep 19, 2024"

#############################
# Get my path i.e CONFIG.py #
#############################
current_file = os.path.abspath(__file__)
path_CONFIG = os.path.dirname(current_file)
sys.path.append(os.path.abspath(path_CONFIG))

'''----------------------------------------------------------------------------
                         USER VARIABLE SETTINGS
----------------------------------------------------------------------------'''

# If machine name is not specified, query the hostname
# This feature gives different names on the nodes. Although the actual name
# can be retrieved by processing the string, it is not advisable.
# So always specift the machine name manually
#if machine is None:
#    machine = socket.gethostname()

# Import the machine configuration
if 'MACHINE_NAME' in os.environ:
    machine_name = os.environ['MACHINE_NAME']
    machine_config = importlib.import_module("machine_config."+machine_name)
else:
    sys.stderr.write("Error: In CONFIG.py'\n")
    sys.stderr.write("       Environment variable 'MACHINE_NAME' not found'\n")
    sys.stderr.write("       Terminating!!!\n")
    exit(1)

######################
# DIRECTORY SETTINGS #
######################
WORK_DIR_BASE = machine_config.WORK_DIR_BASE
SLEEP_TIME = machine_config.SLEEP_TIME

####################################
# CALCULATOR ENVIRONMENT VARIABLES #
####################################


def get_vasp_environ():

    PATH_VASP_STD = machine_config.PATH_VASP_STD
    PATH_VASP_GAM = machine_config.PATH_VASP_GAM
    PATH_VASP_NCL = machine_config.PATH_VASP_NCL
    VASP_MPI_TAGS = machine_config.VASP_MPI_TAGS
    VASP_ENV_TAGS = machine_config.VASP_ENV_TAGS

    return (PATH_VASP_STD, PATH_VASP_GAM, PATH_VASP_NCL, VASP_MPI_TAGS, VASP_ENV_TAGS)

############################
# SLURM SCHEDULER SETTINGS #
############################  
SCHEDULER = machine_config.SCHEDULER
MEM_PER_CPU_MB = machine_config.MEM_PER_CPU_MB
PARTITION = machine_config.PARTITION
QOS = machine_config.QOS
MAX_PROCESSORS_PER_NODE = machine_config.MAX_PROCESSORS_PER_NODE
MAX_NODES_PER_JOB = machine_config.MAX_NODES_PER_JOB
IS_HETEROGENEOUS = machine_config.IS_HETEROGENEOUS

##############################################
# Create work directory if it does not exist #
##############################################
if not os.path.exists(WORK_DIR_BASE):
    os.makedirs(WORK_DIR_BASE)

'''-----------------------------------------------------------------------------
                            END OF MODULE
-----------------------------------------------------------------------------'''
