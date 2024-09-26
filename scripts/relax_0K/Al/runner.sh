#!/bin/bash

APP=../../matkit/apps/app_as_it_is_automator.py
WORKDIR=calculations/relax_0K/Al

# SETUP
SETUP=scripts.relax_0K.Al.setup

# SETUP
if [ $1 == "SETUP" ]; then
    python $APP \
        SETUP --work_dir=$WORKDIR \
              --setup_filename=$SETUP \
              --wall_time=86400 \
              --is_taskfarm=True \
              --taskfarm_n_proc=8 \
              --n_taskfarm=1
fi

# RUN
if [ $1 == "RUN" ]; then
    python $APP \
        RUN --work_dir=$WORKDIR \
            --processing_mode=taskfarm
fi

# PROCESS
if [ $1 == "PROCESS" ]; then
    python $APP \
        PROCESS --work_dir=$WORKDIR
fi

# TREE CLEAN
if [ $1 == "TREE_CLEAN" ]; then
    python $APP \
        TREE_CLEAN --work_dir=$WORKDIR \
                   --clean_level=saver
fi
