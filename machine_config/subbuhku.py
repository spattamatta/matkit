'''-----------------------------------------------------------------------------
                              subbuhku.py

 Description: Machine settings for Subrahmanyam's office desktop at HKU.

-----------------------------------------------------------------------------'''
import os

__author__ = "A. S. L. Subrahmanyam Pattamatta"
__copyright__ = "Copyright 2024, MaTKit"
__version__ = "1.0"
__maintainer__ = "A. S. L. Subrahmanyam Pattamatta"
__email__ = "lalithasubrahmanyam@gmail.com"
__status__ = "Alpha"

'''-----------------------------------------------------------------------------
                                 USER SETTINGS
-----------------------------------------------------------------------------'''

######################
# DIRECTORY SETTINGS #
######################

# All outputs will be stored in the WORK_DIR_BASE / <Relative directory the user sets in the scripts>
WORK_DIR_BASE = '/media/subrahmanyam/ssd2t/WORK'
SLEEP_TIME = 5.0 # When the workflow creates input files and begins a run, some time the filesystem may lag and if the code is run immediately it might throuw a file not found error, so sleep for sometime

#######################
# CALCULATOR SETTINGS #
#######################

# VASP
# NOTE: If you use environment modules to load vasp or have added vasp bin to your default shell environment, you can leave out path_vasp_bin
path_vasp_bin = os.getenv('VASP_BIN')
if path_vasp_bin is None:
    path_vasp_bin = ''
else:
    path_vasp_bin = path_vasp_bin + '/'
PATH_VASP_STD = path_vasp_bin + 'vasp_std' # Required
PATH_VASP_GAM = path_vasp_bin + 'vasp_gam' # Optional: Only if workflow needs gamma point calculations
PATH_VASP_NCL = path_vasp_bin + 'vasp_gam' # Optional: Only if workflow needs non-collinear spin calculations
VASP_MPI_TAGS = None
VASP_ENV_TAGS = ['module load vasp']

############################
# SLURM SCHEDULER SETTINGS #
############################  
SCHEDULER = 'slurm'           # Currently only slurm is supported
MEM_PER_CPU_MB = 3500         # Maximum memory per core that can be used
PARTITION = 'debug'           # Set partition or queue name of the scheduler
QOS = None                    # Set quality of service, can be set to None
MAX_PROCESSORS_PER_NODE = 192 # Maximum processors per node
MAX_NODES_PER_JOB = 1         # Maximum numbe rof nodes that can be used
IS_HETEROGENEOUS = False      # Do all nodes have the same number of cores or not

'''-----------------------------------------------------------------------------
                            END OF CONFIGURATION
-----------------------------------------------------------------------------'''
