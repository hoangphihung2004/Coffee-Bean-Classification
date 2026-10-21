import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is on python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    print("=" * 70)
    print("  COFFEE BEAN EDGE AI INSPECTOR - FASTAPI SERVER")
    print("=" * 70)
    print("  Host:    http://0.0.0.0:8000")
    print("  Local:   http://localhost:8000")
    print("  Docs:    http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
