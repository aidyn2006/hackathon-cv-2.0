#!/usr/bin/env python3
"""Wrapper script to start the Flask app with correct paths"""
import sys
import os

# Add venv site-packages to path
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib', 'python3.13', 'site-packages')
sys.path.insert(0, venv_path)

# Now import and run the app
if __name__ == '__main__':
    # Import the app module
    import app

