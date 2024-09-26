'''----------------------------------------------------------------------------
                             mod_batch.py

 Description: Prepare job submit scripts for various batch systems.

 Author: Subrahmanyam Pattamatta
 Contact: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import math
import json
import shutil
import subprocess
from pathlib import Path
from copy import deepcopy

# Externally installed modules
# None

# Local imports
from matkit import CONFIG
from matkit.calculators import mod_calculator, mod_vasp
from matkit.core import mod_utility, mod_hostlist, mod_constrained_optimization

'''----------------------------------------------------------------------------
                                     MODULE VARIABLES
----------------------------------------------------------------------------'''
module_name = "mod_batch.py"

class Scheduler_Info:
    file_name = None
    job_driver_filename = None
    job_name = None
    email = None
    mail_type = None
    partition = None
    qos = None
    ntasks = None
    cpus_per_task = None
    nnodes = None
    ntasks_per_core = None
    ntasks_per_node = None
    ntasks_per_socket = None
    distribution = None
    mem = None
    mem_per_cpu_mb = None
    time_seconds = None
    stdout_flnm = None
    stderr_flnm = None
    exe = None

    # parameterized constructor
    def __init__(self,
                 file_name="submit.sh",
                 job_driver_filename="job_driver.py",
                 job_name="MaTKit",
                 email=None,
                 mail_type="ALL",
                 partition=None,
                 qos=None,
                 ntasks=1,
                 cpus_per_task=1,
                 nnodes=None,
                 ntasks_per_core=1,
                 ntasks_per_node=None,
                 ntasks_per_socket=None,
                 distribution=None,
                 mem=None,
                 mem_per_cpu_mb=None,
                 time_seconds=1800.0,
                 stdout_flnm="job.out",
                 stderr_flnm="job.err",
                 exe=None
                 ):
        self.file_name = file_name
        self.job_driver_filename = job_driver_filename
        self.job_name = job_name
        self.email = email
        self.mail_type = mail_type
        self.partition = partition
        self.qos = qos
        self.ntasks = ntasks
        self.cpus_per_task = cpus_per_task
        self.nnodes = nnodes
        self.ntasks_per_core = ntasks_per_core
        self.ntasks_per_node = ntasks_per_node
        self.ntasks_per_socket = ntasks_per_socket
        self.distribution = distribution
        self.mem = mem
        self.mem_per_cpu_mb = mem_per_cpu_mb
        self.time_seconds = time_seconds
        self.stdout_flnm = stdout_flnm
        self.stderr_flnm = stderr_flnm
        self.exe = exe

    def print(self):
        sys.stdout.write("Printing Scheduler_Info object\n")
        if self.file_name is not None:
            sys.stdout.write("file_name : %s\n" % (self.file_name))
        if self.job_driver_filename is not None:
            sys.stdout.write("job_driver_filename : %s\n" % (self.job_driver_filename))
        if self.job_name is not None:
            sys.stdout.write("job_name : %s\n" % (self.job_name))
        if self.email is not None:
            sys.stdout.write("email : %s\n" % (self.email))
        if self.mail_type is not None:
            sys.stdout.write("email : %s\n" % (self.mail_type))
        if self.partition is not None:
            sys.stdout.write("partition : %s\n" % (self.partition))
        if self.qos is not None:
            sys.stdout.write("Quality of service (qos) : %s\n" % (self.qos))
        if self.ntasks is not None:
            sys.stdout.write("ntasks : %d\n" % (self.ntasks))
        if self.cpus_per_task is not None:
            sys.stdout.write("cpus_per_task : %d\n" % (self.cpus_per_task))
        if self.nnodes is not None:
            sys.stdout.write("nnodes : %d\n" % (self.nnodes))
        if self.ntasks_per_core is not None:
            sys.stdout.write("ntasks_per_core : %d\n" % (self.ntasks_per_core))
        if self.ntasks_per_node is not None:
            sys.stdout.write("ntasks_per_node : %d\n" % (self.ntasks_per_node))
        if self.ntasks_per_socket is not None:
            sys.stdout.write(
                "ntasks_per_socket : %d\n" %
                (self.ntasks_per_socket))
        if self.distribution is not None:
            sys.stdout.write("distribution : %s\n" % (self.distribution))
        if self.mem is not None:
            sys.stdout.write("mem : %d\n" % (self.mem))
        if self.mem_per_cpu_mb is not None:
            sys.stdout.write("mem_per_cpu_mb : %d\n" % (self.mem_per_cpu_mb))
        if self.time_seconds is not None:
            sys.stdout.write("time_seconds : %f\n" % (self.time_seconds))
        if self.stdout_flnm is not None:
            sys.stdout.write("stdout_flnm : %s\n" % (self.stdout_flnm))
        if self.stderr_flnm is not None:
            sys.stdout.write("stderr_flnm : %s\n" % (self.stderr_flnm))
        if self.exe is not None:
            sys.stdout.write("exe : %s\n" % (self.exe))

'''----------------------------------------------------------------------------
                                   SUBROUTINES
----------------------------------------------------------------------------'''
##################
### SUBROUTINE ###
##################

def create_hostfile(scheduler=None):

    '''
    Hostfile contains a list of processor names, that will be used by a parallel
    MPI job.
    '''

    mod_utility.error_check_argument_required(
        arg_val=scheduler, arg_name="scheduler", module=module_name,
        subroutine="create_hostfile", valid_args=["slurm"])

    if scheduler == "slurm":
        node_list_string = os.environ['SLURM_JOB_NODELIST']
        tasks_per_node_list_string = os.environ['SLURM_TASKS_PER_NODE']

        node_list = mod_hostlist.expand_hostlist(node_list_string)
        tasks_per_node_list = mod_hostlist.parse_slurm_tasks_per_node(tasks_per_node_list_string)

        with open("hostfile", "w") as fh:
            for host, nprocs in zip(node_list, tasks_per_node_list):
                for i in range(0,nprocs):
                    fh.write("%s\n" %(host))

##################
### SUBROUTINE ###
##################


def get_nodes_ppn(n_proc):

    # If the machine is heterogeneous do not specify nodes and processors per
    # node
    if CONFIG.IS_HETEROGENEOUS:
        return (None, None)

    max_ppn = CONFIG.MAX_PROCESSORS_PER_NODE

    # Sanity check
    if n_proc == 0:
        sys.stderr.write("Error: In module 'mod_batch.py'\n")
        sys.stderr.write("       In subroutine 'get_nodes_ppn'\n")
        sys.stderr.write("       Nonzero number of processors 'n_proc' required\n")
        sys.stderr.write("       Number of processors: n_proc = %d\n" %(n_proc))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Sanity check
    if max_ppn == 0:
        sys.stderr.write("Error: In module 'mod_batch.py'\n")
        sys.stderr.write("       In subroutine 'get_nodes_ppn'\n")
        sys.stderr.write("       Nonzero 'CONFIG.MAX_PROCESSORS_PER_NODE' required\n")
        sys.stderr.write("       Check CONFIG.py file\n")
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Quotient and remainder
    quo = int(n_proc / max_ppn)
    rem = n_proc % max_ppn

    # Exact divisible
    if rem == 0:
        n_nodes = quo
        ppn = max_ppn

    else:
        n_nodes = quo + 1
        # If processors do not fill up even a single node
        if quo == 0:
            ppn = rem
        else:
            ppn = math.ceil(n_proc/n_nodes)

    if n_nodes > CONFIG.MAX_NODES_PER_JOB:
        sys.stderr.write("Error: In module 'mod_batch.py'\n")
        sys.stderr.write("       In subroutine 'get_nodes_ppn'\n")
        sys.stderr.write("       Maximum number of nodes per job exceeded\n")
        sys.stderr.write("       Set at: n_nodes = %d\n" %(n_nodes))
        sys.stderr.write("       Required: n_nodes = %d\n" %(CONFIG.MAX_NODES_PER_JOB))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    return (n_nodes, ppn)

##################
### SUBROUTINE ###
##################


def write_job_driver(work_dir="./"):

    # Create job_driver for processing elastic job
    with open(work_dir + "/job_driver.py", "w") as fh:
        fh.write("import os\n"
                 "import sys\n"
                 "import ntpath\n"
                 "from pathlib import Path\n"
                 "sys.path.append(os.path.abspath('%s'))\n"
                 "current_file = os.path.abspath(__file__)\n"
                 "current_dir = os.path.dirname(current_file)\n"
                 "os.chdir(current_dir)\n"
                 "n_proc = int(sys.argv[1])\n"
                 "if len(sys.argv) == 3:\n"
                 "    hostfile = sys.argv[2]\n"
                 "    hostfile = ntpath.basename(hostfile)\n"
                 "else:\n"
                 "    hostfile = None\n"
                 % (CONFIG.path_CONFIG)
                 )

##################
### SUBROUTINE ###
##################


def mpi_command(n_proc=1, hostfile=None, wdir=None):

    if CONFIG.SCHEDULER == "slurm":
        if hostfile is None:
            mpi_cmd = "mpirun -np " + str(n_proc) + " " #" --oversubscribe "
        else:
            mpi_cmd = "mpirun -np " + str(n_proc) + " --hostfile " + \
                str(hostfile) + " " #" --oversubscribe "

        if wdir is not None:
            mpi_cmd = mpi_cmd + " --wdir " + str(wdir) + " "

        return mpi_cmd       

##################
### SUBROUTINE ###
##################


def n_proc_str():

    if CONFIG.SCHEDULER == "slurm":
        return "$SLURM_NTASKS"


##################
### SUBROUTINE ###
##################


def generate_submit_script(batch_object):

    if CONFIG.SCHEDULER == "slurm":
        generate_slurm_submit_script(batch_object)


##################
### SUBROUTINE ###
##################


def generate_taskfarm_submit_script(
        batch_object, taskfarm_n_proc=None,
        submit_script_filename="taskfarm_submit.sh"):

    batch_object_taskfarm = deepcopy(batch_object)
    if taskfarm_n_proc is not None:
        batch_object_taskfarm.ntasks = taskfarm_n_proc
    batch_object_taskfarm.file_name = submit_script_filename

    # NOTE: This variables has to be calculated automatically depending on the
    #       machine parameters
    is_span_nodes = False

    if CONFIG.SCHEDULER == "slurm":
        generate_slurm_submit_script(batch_object_taskfarm)
        with open(batch_object_taskfarm.file_name, "a") as fh:
            fh.write("result=$(python <<EOF\n"
                     "import os\n"
                     "import sys\n"
                     "sys.path.append(os.path.abspath('%s'))\n"
                     "from matkit.core import mod_batch\n"
                     "mod_batch.create_hostfile('slurm')\n"
                     "EOF\n"
                     ")\n"
                     %(CONFIG.path_CONFIG)
                    )

    # Append taskfarming routine
    with open(batch_object_taskfarm.file_name, "a") as fh:
        fh.write("result=$(python <<EOF\n"
                 "import os\n"
                 "import sys\n"
                 "sys.path.append(os.path.abspath('%s'))\n"
                 "from matkit.utility.pytaskfarm import pytaskfarm\n"
                 "pytaskfarm.run_taskfarm('hostfile', 'taskfarm_jobfile', %s)\n"
                 "EOF\n"
                 ")\n"
                 %(CONFIG.path_CONFIG, is_span_nodes)
                )

##################
### SUBROUTINE ###
##################


def submit_job(script_flnm="submit.sh"):

    if CONFIG.SCHEDULER == "slurm":
        # NOTE: /bin/sh is usually some other shell trying to mimic The Shell.
        # Many distributions use /bin/bash for sh. So explicitly provide executable="/bin/bash"
        # If you do not do this, python will invoke /bin/sh and you might get errors such as module command not found
        # if there is a module load vasp kind of command in the submit script
        process = subprocess.call("sbatch " + script_flnm, shell=True, executable="/bin/bash")

##################
### SUBROUTINE ###
##################


def generate_slurm_submit_script(slurm_object):

    [n_nodes, ppn] = get_nodes_ppn(slurm_object.ntasks)
    slurm_object.nnodes = n_nodes

    fh = open(slurm_object.file_name, "w")
    #fh.write("#!/usr/bin/env bash\n")
    fh.write("#!/bin/bash\n")


    # Set name of job name
    fh.write("#SBATCH --job-name=%s\n" % (slurm_object.job_name))

    # Send mail to user address
    fh.write("#SBATCH --mail-user=%s\n" % (slurm_object.email))

    # Set mail alert at start, end and abortion of execution (NONE, BEGIN,
    # END, FAIL, ALL)
    fh.write("#SBATCH --mail-type=%s\n" % (slurm_object.mail_type))

    # Set partition
    if slurm_object.partition is not None:
        fh.write("#SBATCH --partition=%s\n" % (slurm_object.partition))

    # Set quality of service (qos)
    if slurm_object.qos is not None:
        fh.write("#SBATCH --qos=%s\n" % (slurm_object.qos))

    # Number of MPI ranks = ntasks
    fh.write("#SBATCH --ntasks=%d\n" % (slurm_object.ntasks))

    # Number of cores per MPI rank
    fh.write(
        "#SBATCH --cpus-per-task=%d\n" %
        (slurm_object.cpus_per_task))

    # Number of nodes
    if slurm_object.nnodes is not None:
        fh.write("#SBATCH --nodes=%d\n" % (slurm_object.nnodes))

    # Number of cores per MPI rank
    fh.write(
        "#SBATCH --ntasks-per-core=%d\n" %
        (slurm_object.ntasks_per_core))

    # How many tasks on each node, to limit the number of tasks per node
    if slurm_object.ntasks_per_node is not None:
        fh.write(
            "#SBATCH --ntasks-per-node=%d\n" %
            (slurm_object.ntasks_per_node))

    # How many tasks on each CPU or socket
    if slurm_object.ntasks_per_socket is not None:
        fh.write(
            "#SBATCH --ntasks-per-socket=%d\n" %
            (slurm_object.ntasks_per_socket))

    # Specify alternate distribution methods for remote processes.
    # -m, --distribution=arbitrary|<block|cyclic|plane=<options>[:block|cyclic|fcyclic]>
    if slurm_object.distribution is not None:
        fh.write(
            "#SBATCH --distribution=%s\n" %
            (slurm_object.distribution))

    # Memory per processor
    if slurm_object.mem is not None:
        fh.write("#SBATCH --mem=%dM\n" % (slurm_object.mem))

    # Memory per processor
    if slurm_object.mem_per_cpu_mb is not None:
        fh.write(
            "#SBATCH --mem-per-cpu=%dM\n" %
            (slurm_object.mem_per_cpu_mb))

    # Set maximum wall clock time hrs:min:sec
    fh.write(
        "#SBATCH --time=%s\n" %
        (mod_utility.convert_seconds(
            slurm_object.time_seconds)))

    # Set standard output
    fh.write("#SBATCH --output=%s\n" % (slurm_object.stdout_flnm))

    # Set standard error
    fh.write("#SBATCH --error=%s\n" % (slurm_object.stderr_flnm))

    #fh.write("ulimit -c unlimited\n")
    #fh.write("ulimit -s unlimited\n")

    [PATH_VASP_STD, PATH_VASP_GAM, PATH_VASP_NCL, VASP_MPI_TAGS, VASP_ENV_TAGS] = CONFIG.get_vasp_environ()

    for env_tag in VASP_ENV_TAGS:
        fh.write("%s\n" %(env_tag))
        
    fh.write("\n")
    fh.write("export OMP_NUM_THREADS=1\n")
    fh.write("export OMPI_MCA_btl_openib_allow_ib=1\n")
    fh.write("cd $SLURM_SUBMIT_DIR\n")

    if slurm_object.exe is not None:

        appexe = "nice -n20 " + str(slurm_object.exe)

        # Run the job
        fh.write("mpirun -np %s %s\n" % (n_proc_str(), appexe))

    fh.close()

##################
### SUBROUTINE ###
##################


def create_taskfarm(batch_object, taskfarm_n_proc, abs_job_dir_list,
                    each_job_n_proc_list=None, n_taskfarm=1,
                    work_dir=".", job_driver_filename="job_driver.py",
                    taskfarm_name="TASKFARM", taskfarm_prefix="TASKFARM_"):
                    
    '''
    Given a work directory and the Absolute job directories (each containing a
    job_driver.py) this subroutine setus up a taskfarm.
    
    Sometimes there can be too many jobs to be placed in a single taskfarm, in
    such a case create multiple taskfarms. This is because getting large chunk
    of processors on the clusters for the larger taskfarm is difficult
    compared to getting smaller chunks of processors.
    
    NOTE: taskform_n_proc is the number of processors for each taskfarm
    NOTE: This subroutine is also used for creating merged taskfarms
    '''

    # Get job chunk sizes
    n_jobs = len(abs_job_dir_list)
    chunk_size = n_jobs // n_taskfarm
    chunk_remainder = n_jobs % n_taskfarm

    # Both indices inclusinve
    sidx = 0
    eidx = 0
    
    # Current directory
    old_dir = os.getcwd()
    
    # Sanity check
    if each_job_n_proc_list is not None:
    
        if (len(each_job_n_proc_list) != len(abs_job_dir_list)):
            sys.stderr.write("Error: In module 'mod_batch.py'\n")
            sys.stderr.write("       In subroutine 'create_taskfarm'\n")
            sys.stderr.write("       length of 'each_job_n_proc_list' should be same as 'abs_job_dir_list'\n")
            sys.stderr.write("       Terminating!!!\n")
            exit(1)
    
    taskfarm_abs_dir_list = []
    # Setup each taskfarm chunk by chunk
    for tidx in range(0, n_taskfarm):

        # Ending index of current chunk
        eidx = sidx + chunk_size - 1

        if chunk_remainder > 0:
            eidx = eidx + 1
            chunk_remainder = chunk_remainder - 1

        # Chunk job list
        chunk_abs_job_dir_list = abs_job_dir_list[sidx:eidx+1] # +1 because python does not include last index

        # Chunk taskfarm directory
        tf_dir = work_dir + "/" + taskfarm_name + '_' + str(tidx).zfill(2)
        if not os.path.exists(tf_dir):
            os.makedirs(tf_dir)
        taskfarm_abs_dir_list.append(tf_dir)

        os.chdir(tf_dir)

        generate_taskfarm_submit_script(
            batch_object=batch_object, taskfarm_n_proc=taskfarm_n_proc,
            submit_script_filename=taskfarm_name.lower() + "_submit.sh")

        #-----------------#
        # Create job file #
        #-----------------#
        # If each_job_n_proc_list is specified use it or else use a common number
        # of processors per job as set by the batch_object
        with open("taskfarm_jobfile", "w") as fh:

            if each_job_n_proc_list is not None:

                chunk_each_job_n_proc_list = each_job_n_proc_list[sidx:eidx+1]
 
                for (job_n_proc, abs_job_dir) in zip(chunk_each_job_n_proc_list, chunk_abs_job_dir_list):
                    fh.write("%d %s %s\n" %(job_n_proc, abs_job_dir, job_driver_filename))

            else:

                for abs_job_dir in chunk_abs_job_dir_list:

                    if os.path.isfile(abs_job_dir + '/job_info.json'):
                        with open(abs_job_dir + '/job_info.json') as fh_job_info:
                            job_info = json.load(fh_job_info)
                        fh.write("%d %s %s\n" %(job_info['n_proc'], abs_job_dir, job_driver_filename))
                    else:
                        fh.write("%d %s %s\n" %(batch_object.ntasks, abs_job_dir, job_driver_filename))
 
        # Add taskfarm waiting tag
        open("#" + taskfarm_prefix + "WAITING#", 'a').close()
        
        # Increment for next iteration
        sidx = eidx + 1

        # Move back to where you were
        os.chdir(old_dir)
        
    # Save taskfarm list
    os.chdir(work_dir)
    with open(taskfarm_name.lower() + "_info.json", "w") as fh:
        json.dump({'abs_dir_list' : taskfarm_abs_dir_list}, fh, indent=4,
            cls=mod_utility.json_numpy_encoder)
    os.chdir(old_dir)
    
##################
### SUBROUTINE ###
##################


def merge_taskfarms(work_dir=None, search_dir_list=None, search_prefix="TASKFARM_", 
                    n_merged_taskfarm=1, batch_object=None, 
                    merged_taskfarm_n_proc=None):

    '''
    Merges taskfarms into single or multiple merged taskfarms by search the list
    of search_dir_list and finding if there are task farms in it
    
    The resulting merged tasfarm(s) are created in the work_dir
    '''
            
    # Get a list of all taskfarm directories, each such directory has a joblist
    dir_list = []
    for search_dir in search_dir_list:
        dir_list.extend(mod_utility.get_waiting_dirs(root_dir=search_dir, prefix=search_prefix))
                                            
    # Get the jobs and each_job_n_proc
    abs_job_dir_list = []
    each_job_n_proc_list = []
    
    for tf_dir in dir_list:
        job_file = tf_dir + "/taskfarm_jobfile"
        # Since job files are small in size, dump them at once as opposed
        # to line by line read-write
        fh = open(job_file, "r")
        lines = fh.readlines()
        fh.close()
        for line in lines:
            line_split_list = line.split()
            abs_job_dir_list.append(line_split_list[1])
            each_job_n_proc_list.append(int(line_split_list[0]))
            # The third element is 'job_driver.py', ignore it
            
    create_taskfarm(
        batch_object=batch_object, taskfarm_n_proc=merged_taskfarm_n_proc,
        abs_job_dir_list=abs_job_dir_list,
        each_job_n_proc_list=each_job_n_proc_list,
        n_taskfarm=n_merged_taskfarm,
        work_dir=work_dir, 
        taskfarm_name="MERGED_TASKFARM", taskfarm_prefix="MERGED_TASKFARM_")

##################
### SUBROUTINE ###
##################


def merge_taskfarms_from_dir_list(
        root_dir_list, work_dir, search_prefix="TASKFARM_", batch_object=None,
        merged_taskfarm_n_proc=None):

    '''
    Merges taskfarms from various directories and places it in work_dir
    '''

    # Get a list of all taskfarm directories
    dir_list = []
    for root_dir in root_dir_list:
        dir_list.extend(mod_utility.get_waiting_dirs(root_dir=root_dir, prefix=search_prefix))

    # Create the merged taskfarm directory
    old_dir = os.getcwd()
    os.chdir(work_dir)

    mtf_dir = "MERGED_TASKFARM"
    if not os.path.exists(mtf_dir):
        os.makedirs(mtf_dir)
    os.chdir(mtf_dir)

    # Generate the submit script
    generate_taskfarm_submit_script(
        batch_object=batch_object, taskfarm_n_proc=merged_taskfarm_n_proc,
        submit_script_filename="merged_taskfarm_submit.sh")

    # Create merged job list
    with open("taskfarm_jobfile", "w") as fh_merged_jf:
        for tf_dir in dir_list:
            job_file = tf_dir + "/taskfarm_jobfile"
            # Since job files are small in size, dump them at once as opposed
            # to line by line read-write
            with open(job_file, "r") as fh_jf:
                fh_merged_jf.write(fh_jf.read())


    # Add taskfarm waiting tag
    open("#MERGED_TASKFARM_WAITING#", 'a').close()

    # Move out of the root directory
    os.chdir(old_dir)

##################
### SUBROUTINE ###
##################


def run_batch_jobs(root_dir=None, run_mode=None):

    # Sanity check: root_dir must be specified and exists
    mod_utility.error_check_dir_exists(dirname=root_dir, module=module_name,
                           subroutine="run_batch_jobs")

    # Check if run mode is correct
    mod_utility.error_check_argument_required(
        arg_val=run_mode, arg_name="run_mode", module=module_name,
        subroutine="run_batch_jobs",
        valid_args=["taskfarm", "merged_taskfarm", "batch"])

    # Get current directory
    old_dir = os.getcwd()

    # Process
    if run_mode == "taskfarm":
        dir_list = mod_utility.get_waiting_dirs(root_dir=root_dir, prefix="TASKFARM_")
        for each_dir in dir_list:
            os.chdir(each_dir)
            submit_job(script_flnm="taskfarm_submit.sh")
            os.chdir(old_dir)

    if run_mode == "merged_taskfarm":
        dir_list = mod_utility.get_waiting_dirs(root_dir=root_dir, prefix="MERGED_TASKFARM_")
        for each_dir in dir_list:
            os.chdir(each_dir)
            submit_job(script_flnm="merged_taskfarm_submit.sh")
            os.chdir(old_dir)

    if run_mode == "batch":
        dir_list = mod_utility.get_waiting_dirs(root_dir=root_dir)
        for each_dir in dir_list:
            os.chdir(each_dir)
            submit_job()
            os.chdir(old_dir)

    # Move back
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
    
##################
### SUBROUTINE ###
##################

'''
def setup_basic_dft_calculations(
        work_dir, rel_dir_list, structure_list, calculator_input_list,
        sim_info_list, batch_object, is_taskfarm=False, n_taskfarm=1,
        taskfarm_n_proc=-1):
   
    # Create work dir and move to it
    old_dir = os.getcwd()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    os.chdir(work_dir)
    work_dir = os.getcwd()

    # Write directory list to the work directory
    calc_info = {
        'calculator' : calculator,
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
        generate_submit_script(batch_object=batch_object_local)

        # Append job_driver.py processing call to the slurm file
        with open(batch_object.file_name, 'a') as fh:
            fh.write("python job_driver.py %d\n"  % (sim_info['n_proc']))

        # Create job_driver.py (header)
        write_job_driver(work_dir="./")

        # Append commands specific to sim_type to job_driver.py
        with open("job_driver.py", "a") as fh:
            fh.write("from matkit.calculators import mod_calculator\n"
                     "mod_calculator.run_dft_job"\
                     "(calculator=%s, n_proc=n_proc, is_final_static=%s, hostfile=hostfile,"\
                     " parset_filename=None, n_iter=%d, wdir='%s')\n"
                     %(sim_info['calculator'], sim_info['is_final_static'], sim_info['n_relax_iter'], work_dir+'/'+rel_dir))

        # Add a waiting tag
        open("#WAITING#", 'a').close()
        job_info = {
            'n_proc' : sim_info['n_proc']
        }
        # The purpose of job_info.json is to let the taskfarm know the number of processors needed
        with open('job_info.json', 'w') as fh:
            json.dump(job_info, fh, indent=4, cls=mod_utility.json_numpy_encoder)

        # If constrained cell relaxation
        if (sim_info['copt_input']['mode_id'] is not None) and \
           (sim_info['copt_input']['flags_voigt'] is not None) and \
           (sim_info['copt_input']['target_stress_voigt'] is not None):

            """
            mod_vasp.setup_constrained_relaxation_file(
                work_dir='./',
                mode_id=sim_info['copt_mode_id'],
                flag_voigt_list=sim_info['copt_flags_voigt'],
                value_voigt_list=sim_info['copt_values_voigt'])
            """
            mod_constrained_optimization.setup(work_dir="./", sim_info=sim_info)

        os.chdir(work_dir)

    # Create a task farm file
    if is_taskfarm:
        create_taskfarm(
            batch_object=batch_object,
            n_taskfarm=n_taskfarm,
            taskfarm_n_proc=taskfarm_n_proc,
            abs_job_dir_list=[work_dir + '/' + rel_dir for rel_dir in rel_dir_list],
            work_dir=work_dir, job_driver_filename="job_driver.py")

    # Move back to old directory
    os.chdir(old_dir)
'''
'''----------------------------------------------------------------------------
                               END OF MODULE
----------------------------------------------------------------------------'''
