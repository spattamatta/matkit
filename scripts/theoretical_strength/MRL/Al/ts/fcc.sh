#!/bin/bash

# This is runner to setup all fcc elements
element_list=(Al)
  
if [ $1 == "SETUP_LOAD" ]; then
  for element in ${element_list[@]}; do
    ./runner_individual_material.sh SETUP_LOAD fcc $element
  done
fi

if [ $1 == "RUN_LOAD" ]; then
  for element in ${element_list[@]}; do
    ./runner_individual_material.sh RUN_LOAD fcc $element
  done
fi

if [ $1 == "PROCESS_LOAD" ]; then
  for element in ${element_list[@]}; do
    ./runner_individual_material.sh PROCESS_LOAD fcc $element
  done
fi

if [ $1 == "EXTRACT_RESULTS" ]; then
  for element in ${element_list[@]}; do
    ./runner_individual_material.sh EXTRACT_RESULTS fcc $element
  done
fi


if [ $1 == "PLOT_STRENGTH_LOAD" ]; then
  for element in ${element_list[@]}; do
    ./runner_individual_material.sh PLOT_STRENGTH_LOAD fcc $element
  done
fi
