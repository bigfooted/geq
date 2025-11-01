#!/bin/bash
# Initialize Git repository and push to GitHub

# Set your GitHub username
GITHUB_USERNAME="bigfootedcreate"
REPO_NAME="g-equation-solver-2d"

echo "Initializing Git repository..."

# Initialize git if not already done
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Output files
*.png
*.jpg
*.pdf
*.dat
*.csv

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Test outputs (keep structure, ignore output files)
tests/*/contour_*.png
tests/*/radius_*.png
tests/*/surface_*.png
tests/*/flame_*.png
tests/*/trajectory_*.png
tests/*/error_*.png
tests/*/comparison_*.png
tests/*/diagnostic_*.png
tests/*/reinitialization_*.png
tests/*/profile_*.png
tests/*/temporal_*.png
EOF

# Add remote
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: 2D G-equation solver with complete test suite

- Core solver with Euler and RK2 time schemes
- Improved solver with reinitialization (PDE and Fast Marching)
- Marching squares algorithm for accurate contour extraction
- 5 comprehensive test cases with validation
- Comparison and diagnostic tools
- Complete documentation and README files"

# Push to GitHub
git branch -M main
git push -u origin main

echo "Repository initialized and pushed to GitHub!"
echo "Visit: https://github.com/$GITHUB_USERNAME/$REPO_NAME"