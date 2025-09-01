"""
Dashboard Runner for Probability and Monte Carlo Dashboards
Launches Dash applications with Bootstrap components
"""

import sys
import os
import subprocess
import time
from typing import Dict, Optional

# Dashboard configurations
DASHBOARDS = {
    "1": {
        "name": "Probability Building Blocks",
        "file": "dashboard_1_probability_building_blocks.py",
        "port": 8050,
        "description": "Explore sample spaces, events, and random variables"
    },
    "4": {
        "name": "Monte Carlo π Estimator", 
        "file": "dashboard_4_monte_carlo_pi.py",
        "port": 8051,
        "description": "Estimate π using Monte Carlo method"
    },
    "6": {
        "name": "Brownian Motion Simulator",
        "file": "dashboard_6_brownian_motion.py", 
        "port": 8052,
        "description": "Explore stochastic processes and random walks"
    },
    "7": {
        "name": "Secretary Problem Simulator",
        "file": "dashboard_7_secretary_problem.py",
        "port": 8053,
        "description": "Optimal stopping and decision-making"
    }
}

def print_dashboard_menu() -> None:
    """Print the available dashboard menu."""
    print("\n" + "="*60)
    print("🎲 PROBABILITY & MONTE CARLO DASHBOARDS")
    print("="*60)
    print("Available Dashboards:")
    print("-" * 40)
    
    for key, dashboard in DASHBOARDS.items():
        print(f"{key}. {dashboard['name']}")
        print(f"   📍 Port: {dashboard['port']}")
        print(f"   📝 {dashboard['description']}")
        print()
    
    print("Commands:")
    print("-" * 40)
    print("  <number>  - Launch specific dashboard")
    print("  all       - Launch all dashboards")
    print("  list      - Show this menu")
    print("  quit      - Exit")
    print("="*60)

def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def launch_dashboard(dashboard_key: str) -> Optional[subprocess.Popen]:
    """Launch a specific dashboard."""
    if dashboard_key not in DASHBOARDS:
        print(f"❌ Error: Dashboard '{dashboard_key}' not found.")
        return None
    
    dashboard = DASHBOARDS[dashboard_key]
    file_path = dashboard['file']
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Dashboard file '{file_path}' not found.")
        return None
    
    # Check if port is available
    if not check_port_available(dashboard['port']):
        print(f"❌ Error: Port {dashboard['port']} is already in use.")
        return None
    
    print(f"🚀 Launching {dashboard['name']}...")
    print(f"   📁 File: {file_path}")
    print(f"   🌐 URL: http://localhost:{dashboard['port']}")
    print(f"   📝 {dashboard['description']}")
    
    try:
        # Launch the dashboard using uv run
        process = subprocess.Popen([
            "uv", "run", "python", file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {dashboard['name']} is running!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start {dashboard['name']}")
            print(f"Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error launching {dashboard['name']}: {e}")
        return None

def launch_all_dashboards() -> Dict[str, subprocess.Popen]:
    """Launch all available dashboards."""
    print("🚀 Launching all dashboards...")
    print("This will start multiple servers on different ports.")
    print()
    
    running_processes = {}
    
    for key, dashboard in DASHBOARDS.items():
        process = launch_dashboard(key)
        if process:
            running_processes[key] = process
            print()
        else:
            print(f"⚠️  Skipping {dashboard['name']} due to error.")
            print()
    
    if running_processes:
        print("✅ All available dashboards launched!")
        print("\n🌐 Dashboard URLs:")
        for key, process in running_processes.items():
            dashboard = DASHBOARDS[key]
            print(f"   {dashboard['name']}: http://localhost:{dashboard['port']}")
        
        print("\n💡 Press Ctrl+C to stop all dashboards.")
        
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all dashboards...")
            for key, process in running_processes.items():
                dashboard = DASHBOARDS[key]
                print(f"   Stopping {dashboard['name']}...")
                process.terminate()
                process.wait()
            print("✅ All dashboards stopped.")
    
    return running_processes

def main() -> None:
    """Main function to run the dashboard launcher."""
    print_dashboard_menu()
    
    while True:
        try:
            choice = input("\n🎯 Enter your choice: ").strip().lower()
            
            if choice == 'quit' or choice == 'exit':
                print("👋 Goodbye!")
                break
            elif choice == 'list':
                print_dashboard_menu()
            elif choice == 'all':
                launch_all_dashboards()
                break
            elif choice in DASHBOARDS:
                process = launch_dashboard(choice)
                if process:
                    print("\n💡 Press Ctrl+C to stop the dashboard.")
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping dashboard...")
                        process.terminate()
                        process.wait()
                        print("✅ Dashboard stopped.")
                break
            else:
                print("❌ Invalid choice. Please enter a valid option.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 