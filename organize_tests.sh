#!/bin/bash
# Script to organize test cases into folders

# Create main tests directory
mkdir -p tests

# Test 1: Expanding Circle (Stationary)
mkdir -p tests/01_expanding_circle_stationary
mv test_expanding_circle.py tests/01_expanding_circle_stationary/

# Test 2: Expanding Circle (Moving with Flow)
mkdir -p tests/02_expanding_circle_moving
mv test_expanding_moving_circle.py tests/02_expanding_circle_moving/

# Test 3: Sharp Initial Condition
mkdir -p tests/03_sharp_initial_condition
mv test_expanding_circle_sharp.py tests/03_sharp_initial_condition/

# Test 4: Sharp IC with Improvements
mkdir -p tests/04_sharp_ic_improved
mv test_expanding_circle_sharp_improved.py tests/04_sharp_ic_improved/

# Test 5: Flame Wall Attachment
mkdir -p tests/05_flame_wall_attachment
mv test_flame_wall_attachment.py tests/05_flame_wall_attachment/

# Comparison tools
mkdir -p tests/comparisons
mv compare_time_schemes.py tests/comparisons/
mv compare_initial_conditions.py tests/comparisons/
mv compare_improvements.py tests/comparisons/

# Diagnostic tools
mkdir -p tests/diagnostics
mv test_reinitialization.py tests/diagnostics/
mv diagnose_reinitialization.py tests/diagnostics/
mv diagnose_reinitialization_coarse.py tests/diagnostics/

echo "Test cases organized into folders!"