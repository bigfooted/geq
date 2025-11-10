# Multi-Frequency FTF Test

This test efficiently computes the Flame Transfer Function (FTF) for multiple forcing frequencies using the memory-optimized callback approach.

## Usage

### Basic Run (Default Frequencies)

```bash
python test_ftf_linear_flame_multiple.py
```

Default frequencies: `[5, 10, 15, 20, 25, 50, 100]` Hz

### Custom Frequencies

```bash
python test_ftf_linear_flame_multiple.py frequencies=10,20,30,40,50
```

### With Optional Visualizations

```bash
# Save time series plots for each frequency
python test_ftf_linear_flame_multiple.py save_time_series

# Save contour plots for each frequency
python test_ftf_linear_flame_multiple.py save_contours

# Save both
python test_ftf_linear_flame_multiple.py save_time_series save_contours
```

### Advanced Options

```bash
python test_ftf_linear_flame_multiple.py \
    frequencies=5,10,20,50,100 \
    drop_cycles=2 \
    measure_cycles=10 \
    steps_per_period=20 \
    A=0.05 \
    K=0.1 \
    output=my_results.csv
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `frequencies=f1,f2,...` | Comma-separated list of frequencies (Hz) | `5,10,15,20,25,50,100` |
| `rk3`, `rk2`, `euler` | Time integration scheme | `rk3` |
| `use_reinit` | Enable reinitialization | `False` |
| `steps_per_period=N` | Time steps per oscillation period | `20` |
| `drop_cycles=N` | Cycles to drop before measurement | `1` |
| `measure_cycles=N` | Cycles to measure for FTF | `10` |
| `A=value` | Forcing amplitude | `0.05` |
| `K=value` | Convection coefficient | `0.1` |
| `save_time_series` | Save time series plots | `False` |
| `save_contours` | Save contour plots | `False` |
| `output=filename.csv` | Output CSV filename | `ftf_multiple_results.csv` |

## Output Files

### Always Created:

1. **`ftf_multiple_results.csv`**: CSV table with all results
   - Columns: `frequency_hz`, `omega`, `gain`, `phase_deg`, `dt`, `steps`, `cfl`, `elapsed_time`

2. **`ftf_multiple_summary.png`**: Summary plot with gain and phase vs frequency
   - Two subplots: Gain (top) and Phase (bottom)
   - Logarithmic frequency axis

### Optional (with flags):

3. **`ftf_multi_time_f{freq}Hz.png`**: Time series plots for each frequency (if `save_time_series`)

4. **`ftf_multi_contours_f{freq}Hz.png`**: Contour plots for each frequency (if `save_contours`)

## Performance

The test uses the memory-optimized callback approach:

**Memory savings**: ~40,000× reduction compared to storing full 2D fields
- Only stores scalar flame lengths (~12 KB per frequency)
- Full 2D fields saved only for optional contour plots (6 snapshots per frequency)

**Example timing** (201×201 grid, RK3, CFL=0.5):
- f=10 Hz (6 cycles): ~21s
- f=20 Hz (6 cycles): ~12s
- f=50 Hz (6 cycles): ~5s
- f=100 Hz (6 cycles): ~2.5s

**7 frequencies**: ~5-10 minutes total

## Example Output

```
================================================================================
ALL SIMULATIONS COMPLETED
================================================================================
Total time: 38.60s (0.6 min)
Average time per frequency: 12.87s

Results Summary:
 Freq (Hz)      Omega         Gain    Phase (°)   Time (s)
------------------------------------------------------------
      10.0      62.83     0.124827      -153.70      21.44
      20.0     125.66     0.150181      -159.31      12.19
      50.0     314.16     0.062144        16.81       4.96
```

## Comparison with Single Frequency Test

| Feature | `test_ftf_linear_flame_single.py` | `test_ftf_linear_flame_multiple.py` |
|---------|-----------------------------------|-------------------------------------|
| Purpose | Single frequency analysis | Multi-frequency sweep |
| Visualizations | Extensive (3-4 plots) | Minimal (summary plot only) |
| Contour plots | Always created (6 snapshots) | Optional (flag required) |
| Time series | Always created (detailed) | Optional (flag required) |
| Checkpoints | Saved by default | Not saved |
| Best for | Detailed single-frequency analysis | Parameter sweeps, FTF curves |

## Recommendations

**For quick parameter sweeps:**
```bash
python test_ftf_linear_flame_multiple.py frequencies=5,10,20,50,100
```

**For publication-quality analysis:**
```bash
# Run multiple script for FTF curve
python test_ftf_linear_flame_multiple.py save_time_series

# Then run single script for detailed analysis at key frequencies
python test_ftf_linear_flame_single.py f=20.0 drop_cycles=10 measure_cycles=20
```

**For debugging/validation:**
```bash
# Test with just 2-3 frequencies and visualizations
python test_ftf_linear_flame_multiple.py \
    frequencies=10,50 \
    measure_cycles=5 \
    save_time_series \
    save_contours
```

## Physics Parameters

- **Domain**: Lx=0.12, Ly=0.25 m
- **Grid**: 201×201 (uniform)
- **Flame speed**: S_L=0.4 m/s
- **Mean velocity**: u_mid=1.0 m/s
- **Forcing**: u'(t) = A·sin(ω(t - K·y))
- **CFL condition**: Automatically satisfied (CFL=0.5)

## Notes

- Reinitialization is **disabled by default** as it can cause jumps in the solution for oscillating flames
- The test uses **Godunov gradient** (1st-order upwind) which is stable for flames with cusps
- For smoother flames, consider using WENO5 spatial scheme (requires code modification)
- Memory usage is minimal (~2 MB per frequency) due to on-the-fly computation
