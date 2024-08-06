#!/bin/bash
set -e

# Clean up old builds
rm -rf dist build *.egg-info
find . -type d -name "__pycache__" -exec rm -r {} +
find . -name "*.pyc" -delete

# Install in editable mode without cache
pip install --no-cache-dir -e .

# Build
python setup.py sdist bdist_wheel 

# Upload
twine upload dist/* --verbose
