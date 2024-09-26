#!/bin/bash

APP=../../../../../matkit/apps/app_constrained_optimization.py
WORKDIR=calculations/theoretical_strength/MRL/Al/COPT_0K

# SETUP
SETUP=scripts.theoretical_strength.MRL.Al.COPT_0K.setup

# SETUP
if [ $1 == "SETUP" ]; then
    python $APP \
        SETUP --work_dir=$WORKDIR \
              --setup_filename=$SETUP \
              --wall_time=86400
fi

# RUN
if [ $1 == "RUN" ]; then
    python $APP \
        RUN --work_dir=$WORKDIR \
            --processing_mode=batch
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
