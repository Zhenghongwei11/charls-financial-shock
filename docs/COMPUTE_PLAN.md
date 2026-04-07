# Compute Plan

This pipeline is designed to run on a standard laptop/desktop once the required CHARLS `.dta` files are available locally.

## Expected runtime (ballpark)
- End-to-end `scripts/reproduce_one_click.sh`: typically minutes to tens of minutes depending on I/O and Python environment.

## Memory
- Peak memory depends on the Stata file reads during `scripts/build_extended_long.py`.

## Notes
- If you encounter memory errors, rerun using a machine with more RAM or ensure that no other large processes are running.

