#!/bin/bash
# Full R2 campaign. Run from the paper directory.
set -e
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
unset R2_SMOKE R2_OUT
for st in calibrate feasible classical train evaluate stress corr timing; do
  echo "=== $st $(date)"
  python3 -m sim.r2 $st --workers 20
done
echo "=== ALL DONE $(date)"
