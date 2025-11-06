# Restart test: Linear flame, non-homogeneous flow (three regions)

This test resumes a simulation from the checkpoint produced by `tests/test_linear_flame_nonhom/test_linear_flame_nonhom.py`.

## How it works
- The base test saves a checkpoint: `tests/test_linear_flame_nonhom/linear_flame_nonhom_ckpt.npz`.
- This restart test loads that checkpoint (grid, fields, time, reinit settings) and continues to `t_final = t0 + t_extend`.
- It reuses the same time-dependent velocity updater and region definitions as the base test.

## Usage
1. Generate the checkpoint by running the base test:
   - Python: `tests/test_linear_flame_nonhom/test_linear_flame_nonhom.py` (saves the NPZ at the end)
2. Run the restart test:
   - Python: `tests/test_restart_linear_flame_nonhom/test_restart_linear_flame_nonhom.py`

The test will print quick region stats and performance info, and assert time continuity across the restart.
