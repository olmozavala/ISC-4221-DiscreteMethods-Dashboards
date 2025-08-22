#!/usr/bin/env python3
"""
Launcher script for the Interactive Graph Algorithms Dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path

def main() -> None:
    """Launch the interactive graph dashboard."""
    print("🚀 Starting Interactive Graph Algorithms Dashboard...")
    print("=" * 60)
    
    # Check if we're in the right directory
    dashboard_file = Path("interactive_graph_dashboards.py")
    if not dashboard_file.exists():
        print("❌ Error: interactive_graph_dashboards.py not found in current directory")
        print("Please run this script from the Graphs directory")
        sys.exit(1)
    
    # Check if uv is available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✓ UV package manager found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: UV package manager not found")
        print("Please install UV: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)
    
    # Check if dependencies are installed
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import streamlit, networkx, plotly, numpy, pandas"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ All dependencies are installed")
        else:
            print("⚠️  Installing dependencies...")
            subprocess.run(["uv", "sync"], check=True)
            print("✓ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)
    
    # Launch the dashboard
    print("\n🌐 Launching dashboard...")
    print("The dashboard will open in your web browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Run the dashboard
        subprocess.run([
            "uv", "run", "streamlit", "run", "interactive_graph_dashboards.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
