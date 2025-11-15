#!/usr/bin/env python3
"""Launch script that properly sets up Python path for venv packages"""
import sys
import os
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent.resolve()

# Add venv site-packages to path
venv_site_packages = BASE_DIR / 'venv' / 'lib' / 'python3.13' / 'site-packages'
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))
    print(f"Added to path: {venv_site_packages}")
else:
    print(f"WARNING: venv site-packages not found at {venv_site_packages}")

# Change to the app directory
os.chdir(BASE_DIR)

# Now start the Flask app
print("Starting Flask application...")
print("=" * 80)

try:
    # Import numpy first to avoid conflicts
    import numpy as np
    print(f"✓ NumPy loaded: {np.__version__}")
    
    # Import Flask app components
    from flask import Flask
    print("✓ Flask loaded")
    
    import cv2
    print("✓ OpenCV loaded")
    
    from ultralytics import YOLO
    print("✓ Ultralytics loaded")
    
    print("=" * 80)
    print("All dependencies loaded successfully!")
    print("=" * 80)
    
    # Import and run the main app
    import app
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease ensure all dependencies are installed in venv")
    sys.exit(1)

