import os
import sys

schedule = "slurm"

def mpi_command(n_proc=1):

    if scheduler == "slurm":
        return "mpirun -np " + str(n_proc) + " --oversubscribe "

    elif scheduler == "pbs":
        return "mpirun -np " + str(n_proc) + " "

def make_python_program_file(dir_name, exe):

    with open(dir_name+"/run_me.py", "w") as fh:
        fh.write("import os\n" \
                 "import sys\n" \
                 "import subprocess\n" \
                 "from pathlib import Path\n" \
                 "n_proc = int(sys.argv[1])\n" \
                 "hf = sys.argv[2]\n" \
                 "exe = '%s'\n" \
                 "run_dir = '%s'\n" \
                 "os.chdir(run_dir)\n" \
                 "process = subprocess.call('mpirun -np ' + str(n_proc) + ' --hostfile ' + hf + ' ' + exe, shell=True)\n" %(exe, dir_name)
                 )

if __name__ == "__main__":

    current_file = os.path.abspath(__file__)
    current_dir =  os.path.dirname(current_file)

    exe = current_dir + "/simple_task/simple_task"
    work_dir = current_dir + "/example_work"

    # Create work directory
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    # Create 4 directories in the WORK directory, each with input files
    n_cases = 4

    fjob = open("jobfile", "w")
    for i in range(1, n_cases + 1):
        case_dir = work_dir+"/CASE_"+str(i)
        if not os.path.isdir(case_dir):
            os.makedirs(case_dir)
        n_proc = i
        make_python_program_file(case_dir, exe)

        fjob.write("%d %s run_me.py\n" %(n_proc, case_dir))

    fjob.close()


        
