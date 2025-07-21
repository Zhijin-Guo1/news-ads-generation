#!/usr/bin/env python3
"""
Launch script for the Streamlit News-Responsive Ad Generator
"""

import subprocess
import sys
import os

def main():
    print("🎯 Starting News-Responsive Ad Generator...")
    print("📍 Make sure you have your OpenAI API key ready!")
    print("🌐 The app will open in your default browser")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            "streamlit", "run", "streamlit/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()