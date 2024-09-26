#!/bin/bash

#------------------------------------------------------------------------------
# Description: This is the runner for setting up ideal strength calculations
# for an individual material.
#
# Author: Subrahmanyam Pattamatta
# Contact: lalithasubrahmanyam@gmail.com
#------------------------------------------------------------------------------

# Apps used
APP=../../../../../matkit/apps/app_incr_load.py

# Directory names based on second input i.e. the element name
if [ "$#" -ne 3 ]; then
    echo "Error: Required enter 3 command line arguments"
    echo "       Argument 1: {SETUP_LOAD, RUN_LOAD, PROCESS_LOAD, SETUP_FORCE_CONSTANTS, RUN_FORCE_CONSTANTS,  PROCESS_FORCE_CONSTANTS}"
    echo "       Argument 2: {Relative_Directory_Append}"
    echo "       Argument 3: {Element_name}"
    echo
    exit 1
fi

# Setup file for stresses
SETUP=scripts.theoretical_strength.MRL.Al.ts.setup
WORK_DIR=calculations/theoretical_strength/MRL/ts/$2/$3

#-----------------------------------------------------------------------------#
#                              LOAD CONTINUATION                              #
#-----------------------------------------------------------------------------#
# NOTE: For all setup runs use workdir as the base dir of the element but for
#   run and process runs use the corresponding LOAD and FORCE_CONSTANTS dir

###############################
### SETUP LOAD CONTINUATION ###
###############################
if [ $1 == "SETUP_LOAD" ]; then
    python $APP \
        SETUP_LOAD --job_name=ts_$3 \
                   --work_dir=$WORK_DIR \
                   --wall_time=400000 \
                   --setup_filename=$SETUP \
                   --is_taskfarm=True \
                   --n_taskfarm=1 \
                   --taskfarm_n_proc=192 \
                   --arg_list=$3
fi

#############################
### RUN LOAD CONTINUATION ###
#############################
if [ $1 == "RUN_LOAD" ]; then
    python $APP \
        RUN --work_dir=$WORK_DIR/LOAD \
            --processing_mode=taskfarm
fi

#################################
### PROCESS LOAD CONTINUATION ###
#################################
if [ $1 == "PROCESS_LOAD" ]; then
    python $APP \
        PROCESS --work_dir=$WORK_DIR/LOAD \
                --calculation_type=LOAD
fi

###########################################
### EXTRACT MULTIPLE LOAD CONTINUATIONS ###
###########################################
if [ $1 == "EXTRACT_RESULTS" ]; then
    python $APP \
        EXTRACT_RESULTS --work_dir=$WORK_DIR/LOAD \
                        --calculation_type=LOAD
fi

################################
### PLOT STRENGTH LOAD BASED ###
################################
if [ $1 == "PLOT_STRENGTH_LOAD" ]; then
    python $APP \
        PLOT_STRENGTH --work_dir=$WORK_DIR/LOAD \
                      --calculation_type=LOAD
fi

#-----------------------------------------------------------------------------#
#                                FORCE CONSTANTS                              #
#-----------------------------------------------------------------------------#

############################
### SETUP FORCE CONSTANT ###
############################
if [ $1 == "SETUP_FORCE_CONSTANTS" ]; then
    python $APP \
        SETUP_FORCE_CONSTANTS --job_name=ids_$3 \
                              --work_dir=$WORK_DIR \
                              --wall_time=200000 \
                              --is_taskfarm=True \
                              --n_taskfarm=1 \
                              --taskfarm_n_proc=64
fi

###########################
### RUN FORCE CONSTANTS ###
###########################
if [ $1 == "RUN_FORCE_CONSTANTS" ]; then
    python $APP \
        RUN --work_dir=$WORK_DIR/FORCE_CONSTANTS
fi

###############################
### PROCESS FORCE CONSTANTS ###
###############################
if [ $1 == "PROCESS_FORCE_CONSTANTS" ]; then
    python $APP \
        PROCESS --work_dir=$WORK_DIR/FORCE_CONSTANTS \
                --calculation_type=FORCE_CONSTANTS
fi

##################################
### PLOT STRENGTH PHONON BASED ###
##################################
if [ $1 == "PLOT_STRENGTH_PHONON" ]; then
    python $APP \
        PLOT_STRENGTH --work_dir=$WORK_DIR/LOAD \
                      --calculation_type=PHONON
fi

#-----------------------------------------------------------------------------#
#                                 END OF SCRIPT                               #
#-----------------------------------------------------------------------------#
