'''----------------------------------------------------------------------------
                                 pytaskfarm.py

 Description: A simple taskfarm implementation. Given a set of hosts and jobs,
              as files, the hostfile and the jobfile, runs the jobs on the
              hosts. Each job can run on a single or multiple cores and in the
              case of multiple cores per job, one can choose to confine the
              alloted cores to the job be on a single node or spread across
              multiple nodes.
              
              Running taskfarm needs a 'hostfile' and a 'jobfile' as explained.

 Example of hostfile: A hostfile is a text file, with its contents as follows:
 
     compute-1
     compute-1
     compute-1
     compute-1
     compute-2
     compute-2
     compute-3
     compute-3
     ... so on.

     Each hostname is repeated as many number of times as the number of cores
     alloted on that particular node. Here the node 'compute-1' has 4 cores,
     the node 'compute-2' has 2 cores and so on.

     NOTE: Hostnames corresponding to different cores on the same node need not
     appear consecutively, as this list will anyways be sorted.

 Example of jobfile: A job file is a text file, with its contents as follows:

     4  /a/b/c    run_code_a.py  args...(if any)
     8  /w/x/y/z  run_code_b.py  args...(if any)
     ... so on.

     Each row is a job with notation as follows:
     <no. of processors for the job> <directory> <python job code>
     <arguments to python job code if any>

     NOTE: The user should make sure that a job does not appear in the job
           file more than once.

 Author: Subrahmanyam Pattamatta
 Email: lalithasubrahmanyam@gmail.com
----------------------------------------------------------------------------'''
# Standard python imports
import os
import sys
import time
import logging
import argparse
import subprocess

# Externally installed modules
# None

# Local imports
import CONFIG
from matkit.core import mod_utility

'''----------------------------------------------------------------------------
                                 MODULE VARIABLES
----------------------------------------------------------------------------'''
# Sleep time for master process when all spawned workers subprocesses are
# either busy doing their assigned job or waiting for few workers to return
# to wind up the taskfarm.
master_sleep_time = 30 # Seconds

# Class holder for a host


class Host_Info:
    name = None      # Name of the host
    n_tot = None     # Total number of processors in the host
    n_free = None    # Number of free processors
    n_busy = None    # Number of processors that are busy working on a job
    n_dead = None    # Number of processors that are unresponsive

# Class holder for a job


class Job_Info:
    job_dir = None   # Directory where the job file is located
    job_file = None  # Python job file
    job_original_idx = None # Actual index of the job as per the jobfile
    status = None    # "WAITING", "PROCESSING", "DONE", "ABORTED", "ERROR"
    n_proc = None    # Number of processors requested for the job
    phandle = None   # Process id if status="PROCESSING"
    ret_code = None  # Return code of the process
    hosts = None     # Each row has [name of host, # of procs of host type]


'''----------------------------------------------------------------------------
                                 SUBROUTINES
----------------------------------------------------------------------------'''
##################
### SUBROUTINE ###
##################


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##################
### SUBROUTINE ###
##################


def sort_jobs(jobs):
    '''
    Sorts jobs in the ascending order of n_proc field
    '''

    jobs.sort(key=lambda x: x.n_proc)

##################
### SUBROUTINE ###
##################


def sort_hosts(hosts):
    '''
    Sorts hosts in the decreasing order of n_free hosts field
    '''

    hosts.sort(key=lambda x: x.n_free, reverse=True)

##################
### SUBROUTINE ###
##################


def find_host(hosts, name):
    '''
    Returns the index of the host if found else -1
    '''

    host_idx = -1

    for i, host in enumerate(hosts):
        if host.name == name:
            host_idx = i
            break

    return host_idx

##################
### SUBROUTINE ###
##################


def count_free_hosts(hosts, is_span_nodes=False):
    '''
    Counts the number of free host either across all the nodes if is_span_nodes
    is True or just the maximum available processors on a single host
    '''

    # Sort hosts in descending order of n_free fields
    sort_hosts(hosts)

    # If spanning all nodes, accumulate all n_free fields
    if is_span_nodes:
        n_free_all = 0
        for host in hosts:
            n_free_all = n_free_all + host.n_free
        return n_free_all

    else:
        # Since it is sorted in descending order, the first element has the
        # largest number of free hosts
        return hosts[0].n_free

##################
### SUBROUTINE ###
##################


def release_hosts(rel_hosts, hosts):
    '''
    returns rel_hosts to the list of hosts
    rel_hosts is an array with each row as [name of host, number of proc]
    '''

    for rel_host in rel_hosts:
        is_inserted = False
        for host in hosts:
            if host.name == rel_host[0]:
                host.n_free = host.n_free + rel_host[1]
                is_inserted = True
                break
        if not is_inserted:
            sys.stderr.write("Error: In module pytaskfarm.py'\n")
            sys.stderr.write("       In subroutine, 'release_hosts'\n")
            sys.stderr.write("       Could not release host %s.\n"
                % (host[0]))
            sys.stderr.write("       Terminating!!!\n")
            exit(1)

##################
### SUBROUTINE ###
##################


def fetch_hosts(hosts, n_hosts_req, is_span_nodes=False):
    '''
    Fetches a given number of hosts from the hosts list, if available
    '''

    # Initialize
    # This will be populated by host dat. Each row has [hostname, # of procs]
    ret_hosts = []

    # Count free hosts, does a descending order sort based on  n_free fields
    n_free_hosts = count_free_hosts(hosts, is_span_nodes=is_span_nodes)
    if n_free_hosts < n_hosts_req:
        return ret_hosts

    # Single node case
    if not is_span_nodes:
        ret_hosts.append([hosts[0].name, n_hosts_req])
        hosts[0].n_free = hosts[0].n_free - n_hosts_req
        return ret_hosts

    # Multiple node spanning case
    else:
        n_hosts_req_counter = n_hosts_req
        for i in range(0, len(free_hosts)):
            if hosts[i].n_free > n_hosts_req_counter:
                ret_hosts.append([hosts[i].name, n_hosts_req_counter])
                free_hosts[i].n_free = free_hosts[i].n_free - \
                    n_hosts_req_counter
                n_hosts_req_counter = 0
            else:
                ret_hosts.append([hosts[i].name, hosts[i].n_free])
                n_hosts_req_counter = n_hosts_req_counter - hosts[i].n_free
                hosts[i].n_free = 0

            if n_hosts_req_counter == 0:
                break

        return ret_hosts

##################
### SUBROUTINE ###
##################


def check_jobs(jobs, hosts, tflog=None):
    '''
    Checks if any jobs are done. If so releases the host resources and updates
    the job status
    '''

    for job_idx, job in enumerate(jobs):
        # If the job is running
        if job.status == "PROCESSING":
            # If the job has terminated
            if job.phandle.poll() is not None:
                # Set return code as status
                job.ret_code = job.phandle.returncode
                if job.ret_code == 0:
                    job.status = "DONE"
                    if tflog is not None:
                        tflog.info("Job #: %d (original #: %d) is processed, "\
                                   "successfully"
                                   % (job_idx+1, job.job_original_idx))
                else:
                    job.status = "ERROR"
                    if tflog is not None:
                        tflog.info("Job #: %d (original #: %d) is processed, "\
                                   "but failed!"
                                   % (job_idx+1, job.job_original_idx))

                # Release resources to hosts
                release_hosts(job.hosts, hosts)

    # Return termination call if all jobs are processed
    are_all_done = True
    for job in jobs:
        if job.status == "WAITING" or job.status == "PROCESSING":
            are_all_done = False

    return are_all_done

##################
### SUBROUTINE ###
##################


def run_job(job_idx, jobs, hosts, is_span_nodes):
    '''
    proc_handle = subprocess.Popen(
    [runner]+runner_args+[exe],
    stdin=subprocess.DEVNULL,
     shell=False)
    return proc_handle
    '''

    # Check if the job directory exists
    if not os.path.isdir(jobs[job_idx].job_dir):
        sys.stderr.write("Error: In module pytaskfarm.py'\n")
        sys.stderr.write("       In subroutine, 'run_job'\n")
        sys.stderr.write("       Job directory '%s' does not exist.\n"
            % (jobs[job_idx].job_dir))
        sys.stderr.write("       Terminating!!!\n")
        exit(1)

    # Fetch resources and create a hostname file
    ret_hosts = fetch_hosts(hosts=hosts,
                            n_hosts_req=jobs[job_idx].n_proc,
                            is_span_nodes=is_span_nodes)
    jobs[job_idx].hosts = ret_hosts
    jobs[job_idx].status = "PROCESSING"

    # Hostfile
    hostfile = jobs[job_idx].job_dir + "/hostfile_" + str(job_idx)
    with open(hostfile, "w") as fh:
        for host in ret_hosts:
            for i in range(0, host[1]):
                fh.write("%s\n" % (host[0]))

    # Run the job as a subproceess in the background
    command = "python " + jobs[job_idx].job_dir + "/" + \
              jobs[job_idx].job_file + " " + str(jobs[job_idx].n_proc) + \
              " " + hostfile
    command_list = command.split()
    jobs[job_idx].phandle = subprocess.Popen(command_list)

##################
### SUBROUTINE ###
##################


def run_taskfarm(hostfile, jobfile, is_span_nodes=False,
                 logfile="taskfarm.log"):
    '''
    This is the main taskfarm routine.
    '''

    # local variables to hold Host_Info objects and Job_Info objects
    hosts=[]
    jobs=[]

    # Create logger
    tflog = mod_utility.create_logger(
        log_object_name='taskfarm', log_file_name=logfile, is_log_console=True,
        file_log_level="DEBUG", console_log_level="ERROR")

    # Welcome message to serve as time stamp
    start_time = time.monotonic() 
    tflog.info("Beginning taskfarm")

    # Check if there is a #TASKFARM_WAITING# tag file
    if os.path.isfile("#TASKFARM_WAITING#"):
        os.rename("#TASKFARM_WAITING#", "#TASKFARM_PROCESSING#")

    ########################
    # Read and setup hosts #
    ########################
    if os.path.isfile(hostfile):
        # Reads hosts into list
        with open(hostfile) as f:
            host_list=f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            host_list=[x.strip() for x in host_list]
            host_list.sort()

    else:
        tflog.error("In module 'pytaskfarm.py'\n" \
                    "In subroutine 'run_taskfarm'\n" \
                    "hostfile '%s' does not exist\n" \
                    "Terminating!!!\n" %(hostfile))
        exit(1)

    # Create hosts object
    for name in host_list:
        host_idx=find_host(hosts, name)
        if host_idx >= 0:
            hosts[host_idx].n_tot=hosts[host_idx].n_tot + 1
            hosts[host_idx].n_free=hosts[host_idx].n_free + 1
        else:
            hosts.append(Host_Info())
            hosts[-1].name=name
            hosts[-1].n_tot=1
            hosts[-1].n_free=1
            hosts[-1].n_busy=0
            hosts[-1].n_dead=0

    #######################
    # Read and setup jobs #
    #######################
    if os.path.isfile(jobfile):
        # Reads hosts into list
        with open(jobfile) as f:
            job_list=f.readlines()
            # remove whitespace characters like `\n` at the end of each line
            job_list=[x.strip() for x in job_list]

    else:
        tflog.error("In module 'pytaskfarm.py'\n" \
                    "In subroutine 'run_taskfarm'\n" \
                    "jobfile '%s' does not exist\n" \
                    "Terminating!!!\n" %(jobfile))
        exit(1)

    # Create jobs object
    for job_original_idx, job_str in enumerate(job_list, start=1):
        job=job_str.split()
        jobs.append(Job_Info())
        jobs[-1].n_proc=int(job[0])
        jobs[-1].job_dir=job[1]
        jobs[-1].job_file=job[2]
        jobs[-1].status="WAITING"
        jobs[-1].job_original_idx = job_original_idx

    ################
    # Control loop #
    ################

    # Jobs are sorted in the ascending order of their resource requirements
    sort_jobs(jobs)

    # Submit the waiting jobs
    for job_idx, job in enumerate(jobs):
        # Wait intil resources are available
        while count_free_hosts(hosts, is_span_nodes) < job.n_proc:
            tflog.info("Waiting for resources to submit job #: %d "\
                       "(original #: %d)" %(job_idx+1, job.job_original_idx))
            time.sleep(master_sleep_time)
            check_jobs(jobs, hosts, tflog)

        # Submit job
        tflog.info("Submitting job #: %d (original #: %d)"
                   %(job_idx+1, job.job_original_idx))
        run_job(job_idx, jobs, hosts, is_span_nodes)

    # Waiting for jobs to finish
    while check_jobs(jobs, hosts, tflog) is False:
        tflog.info("Waiting for jobs to finish")
        time.sleep(master_sleep_time)

    # Check if there is a #TASKFARM_PROCESSING# tag file
    if os.path.isfile("#TASKFARM_PROCESSING#"):
        os.rename("#TASKFARM_PROCESSING#", "#TASKFARM_DONE#")

    # Closing message
    end_time = time.monotonic()
    run_time_str = mod_utility.convert_seconds(end_time - start_time)
    tflog.info("Total run time HH:MM:SS : %s" %(run_time_str))
    tflog.info("Ending taskfarm")

    # FUTURE WORK:
    # 1. Error handling: Mark hosts as dead that are unresponsive at all stages
    #    of taskfarm.
    # 2. Any hosts that die during processing. The job associated with that
    #    set of hosts has to be marked as error, so that future restarts can
    #    preocess them.
    # 3. Any processes that do not return in time. How does the task farm know
    #    the assigned wall time by the submit script? May be pass as an
    #    argument to the taskfarm.
    # 4. Looking at the log file, from the time of submission to the return
    #    of a job, the user can calculate the time taken by a job (with
    #    accuracy +/- master_sleep_time). Let the taskfarm itself compute the
    #    time taken by each job and print it to the log file at the end.
    # 5. Initially the jobs are renumbered in ascending order of the number of
    #    of processors requested by the job. This renumbering has to be mapped
    #    back to the original numbering when printing any job related
    #    information.
    # 6. Periodically print statistics # jobs processed, # Waiting in queue,
    #    # Processed and failed # Processed successfully
    # 7. The taskfarm should have ways be aware of the time limits set by the
    #    scheduler. It should have total_wall_time, buffer_time (to allow jobs
    #    to return and not submit any new jobs), abort any running jobs if
    #    in the critical zone (typically a few seconds before the completion of
    #    the total_wall_time and place #ABORTED# flags,for restarting next time
    # 8. Release idling resources back to the scheduler ? Far fetched idea ?
    # 9. The taskfarm showld periodically read a COMMUNICATOR.json file.
    #    This serves as a way for the user to communicate with the taskfarm 
    #    while the job is running.
    #    The comminocator should be able to coney the following to the
    #    taskfarm:
    #    (a). Hard abort the jobs, communicate to the deamon
    #    (b). Stop submitting any new jobs, communiate to the deamon
    #    (c). Read additional jobfile to add new jobs to the taskfarm, communicate file to deamon (allow something like git add, commit and push, view status (handshake))
    #    (d). 
    # 10. Safe locks while opening and closing communicators
    # 11. Job dependencies for pre-existing as well as on the fly creation such as run C after A and B have comleted etc.
    # 12. Multiple farmers

'''----------------------------------------------------------------------------
                                     MAIN
----------------------------------------------------------------------------'''
if __name__ == "__main__":

    ##########
    # Parser #
    ##########

    parser=argparse.ArgumentParser(usage='Task-farmer for python.')

    parser.add_argument('-hf', '--hostfile', type=str, help='Enter the host' \
        ' filename. Each line is a core or a host.', required=True,
        action='store', dest='hostfile')

    parser.add_argument('-jf', '--jobfile', type=str, help='Enter the job' \
        ' filename. Each line is a python command and is counted as a job.' \
        ' It is the responsibility of the user to make sure no jobs are' \
        ' repeated etc. The taksfarm blindly submits the jobs in the jobfile',
        required=True, action='store', dest='jobfile')

    parser.add_argument(
        '--span-nodes', '-sn', type=str2bool,
        help='Specify if all the processors assigned to a particular job have \
        to be on the same node or can span they multiple nodes. The deafult \
        value is False meaning that all cores assigned to a job should be \
        from the same node.', default=False, action='store',
        dest='is_span_nodes')

    parser.add_argument(
        '--logfile', '-lf', type=str, help='Specify the name of the logfile.',
        default='taskfarm.log', action='store', dest='logfile')

    args=parser.parse_args()

    #################
    # End of parser #
    #################

    run_taskfarm(args.hostfile, args.jobfile, args.is_span_nodes, args.logfile)

'''----------------------------------------------------------------------------
                                 END OF MAIN
----------------------------------------------------------------------------'''
