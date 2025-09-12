#!/usr/bin/env python3
"""
ML Recommendation Service Startup Script
This script starts the FastAPI ML service for internship recommendations.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pandas
        import sklearn
        import fastapi
        import uvicorn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if required data files exist."""
    data_dir = Path(__file__).parent / "data"
    required_files = ["internships.csv", "candidates.csv"]
    
    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"❌ Missing data file: {file_path}")
            return False
        print(f"✅ Found data file: {file_path}")
    
    return True

def start_ml_service(port=8000, host="0.0.0.0"):
    """Start the ML recommendation service."""
    print(f"🚀 Starting ML Recommendation Service on {host}:{port}")
    
    # Change to the ML model directory
    ml_dir = Path(__file__).parent
    os.chdir(ml_dir)
    
    try:
        # Start the FastAPI service
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", host, 
            "--port", str(port),
            "--reload"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 ML service stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start ML service: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_ml_service(port=8000):
    """Test if the ML service is responding."""
    try:
        response = requests.get(f"http://localhost:{port}/", timeout=5)
        if response.status_code == 200:
            print("✅ ML service is responding")
            return True
        else:
            print(f"❌ ML service returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ML service not responding: {e}")
        return False

def main():
    """Main function to start the ML service."""
    print("🤖 ML Recommendation Service Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Get port from environment or use default
    port = int(os.environ.get("ML_SERVICE_PORT", 8000))
    host = os.environ.get("ML_SERVICE_HOST", "0.0.0.0")
    
    print(f"\n📡 Service Configuration:")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   API URL: http://{host}:{port}")
    print(f"   Docs: http://{host}:{port}/docs")
    
    # Start the service
    print(f"\n🚀 Starting service...")
    start_ml_service(port, host)

if __name__ == "__main__":
    main()
