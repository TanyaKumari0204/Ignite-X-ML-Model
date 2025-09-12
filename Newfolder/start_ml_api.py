#!/usr/bin/env python3
"""
Startup script for ML Recommendation API
This script starts the FastAPI server for the internship recommendation system.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Start the ML API server"""
    print("🚀 Starting ML Recommendation API...")
    print("📊 Loading internship data and training models...")
    
    # Check if data file exists
    data_file = current_dir / "data" / "internships.csv"
    if not data_file.exists():
        print(f"❌ Error: Data file not found at {data_file}")
        print("Please ensure the internships.csv file exists in the data/ directory")
        sys.exit(1)
    
    print(f"✅ Data file found: {data_file}")
    print("🤖 Initializing recommendation engine...")
    
    try:
        # Import and test the recommendation function
        from notebooks.recommend import recommend_for_candidate
        
        # Test with sample data
        test_candidate = {
            "education": "Computer Science",
            "skills": "Python, Machine Learning, Data Analysis",
            "interests": "Technology, Innovation",
            "preferred_location": "Remote",
            "mode": "Remote",
            "top_n": 3
        }
        
        test_recommendations = recommend_for_candidate(test_candidate)
        print(f"✅ Recommendation engine working! Generated {len(test_recommendations)} test recommendations")
        
    except Exception as e:
        print(f"❌ Error initializing recommendation engine: {e}")
        sys.exit(1)
    
    # Start the server
    print("🌐 Starting FastAPI server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 ML API server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
