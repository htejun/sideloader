#!/bin/bash

set -xe
cd "$(dirname $0)"

# Tunables
MBS=100  # MB/s

# Constants
TOTAL_RUNAWAY=$((128<<30))  # 128GB
FRAC_PER_LOOP=$(echo "${MBS} * 1024 * 1024 / ${TOTAL_RUNAWAY}" | bc -l)

# Run runaway process
env ALLOCATION="${TOTAL_RUNAWAY}" FRAC_PER_LOOP="${FRAC_PER_LOOP}" ./dlxu-memory-growth.py
