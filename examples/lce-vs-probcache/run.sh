#!/bin/sh

# Enable command echo
set -v

# Directory where this script is located
CURR_DIR=`pwd`

# Icarus main folder
ICARUS_DIR=${CURR_DIR}/../..

# Config file
CONFIG_FILE=${CURR_DIR}/config.py

# Pickle file where results will be saved
RESULTS_FILE=${CURR_DIR}/results.pickle

# Output text file by converting results.pickle into human-readable format
TEXT_FILE=${CURR_DIR}/results.txt

# Add Icarus code to PYTHONPATH
export PYTHONPATH=${ICARUS_DIR}:$PYTHONPATH

# Run experiments
echo "Run experiments"
icarus run --results ${RESULTS_FILE} ${CONFIG_FILE}
icarus results print ${RESULTS_FILE} ${TEXT_FILE}