#!/bin/sh

# Enable command echo
set -v

# Directory where this script is located
CURR_DIR=`pwd`

# Icarus main folder
ICARUS_DIR=${CURR_DIR}/../..

# Config file
CONFIG_FILE=${CURR_DIR}/config.py

# File where results will be saved
RESULTS_FILE=${CURR_DIR}/results.pickle

# Add Icarus code to PYTHONPATH
export PYTHONPATH=${ICARUS_DIR}:$PYTHONPATH

# Run experiments
echo "Run experiments"
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/icarus run --results ${RESULTS_FILE} ${CONFIG_FILE}
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/icarus results print results.pickle > results.txt
