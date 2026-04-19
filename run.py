"""
run.py — Launch both FastAPI backend and Streamlit frontend together.

Usage:
    python run.py

Press Ctrl+C to shut down both servers cleanly.
"""

import subprocess
import sys
import time
import os


def main():
    print("Starting FastAPI backend on http://127.0.0.1:8000 ...")

    # Resolve the python executable inside the active venv
    python = sys.executable

    # ── Start FastAPI (uvicorn) ───────────────────────────────────────────────
    backend = subprocess.Popen(
        [python, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    # Give the backend a few seconds to initialize before opening the UI
    print("Waiting for backend to initialize...")
    time.sleep(5)

    print("Starting Streamlit frontend on http://localhost:8501 ...")

    # ── Start Streamlit ───────────────────────────────────────────────────────
    frontend = subprocess.Popen(
        [python, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )

    print("\nBoth servers are running.")
    print("  Backend  → http://127.0.0.1:8000")
    print("  Frontend → http://localhost:8501")
    print("\nPress Ctrl+C to stop both servers.\n")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend.terminate()
        frontend.terminate()
        backend.wait()
        frontend.wait()
        print("Done.")


if __name__ == "__main__":
    main()