#!/usr/bin/env python3
"""
Entry point for the Streamlit News-Responsive Ad Generator
This file is used by cloud platforms that expect app.py as the entry point
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main Streamlit app
from streamlit.streamlit_app import *

if __name__ == "__main__":
    # This will be handled by streamlit run app.py
    pass