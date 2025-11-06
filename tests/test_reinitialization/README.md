# Reinitialization Comparison

Compare global vs local (narrow-band) reinitialization strategies for the level set.

Setup
- Expanding circle with S_L = 0.2, u = (0,0)
- Sharp initial condition (inside=-1, outside=+1)
- Multiple configurations: none, smooth IC only, Fast March (global/local, varying intervals), PDE (global/local)

Outputs
- Error evolution (linear and log) vs time
- Summary bar charts: max/mean/final error and wall time per configuration
- Console summary table with key statistics and direct LOCAL vs GLOBAL comparisons

Run
- python test_reinitialization.py
- python test_reinitialization.py t=2.0

Notes
- LOCAL reinitialization is the default and recommended approach
- Smooth initial condition further improves accuracy and robustness
