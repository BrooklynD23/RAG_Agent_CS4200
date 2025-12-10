#!/usr/bin/env python3
"""Utility to verify dependencies, initialize vector DB, and launch backend + frontend together.

This script handles:
- Installing Python dependencies from requirements.txt
- Initializing the ChromaDB vector store directory
- Starting the FastAPI backend with RAG endpoints
- Starting the Streamlit frontend
- Graceful shutdown of all services
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = ROOT_DIR / "requirements.txt"
STREAMLIT_APP = ROOT_DIR / "src" / "news_rag" / "ui" / "streamlit_app.py"
UVICORN_APP = "src.news_rag.api.server:app"
CHROMA_DIR = ROOT_DIR / ".chroma_db"


def ensure_dependencies(skip_install: bool, upgrade: bool = False) -> None:
    """Ensure Python dependencies defined in requirements.txt are installed.
    
    Args:
        skip_install: If True, skip dependency installation entirely.
        upgrade: If True, upgrade packages to latest versions.
    """

    if skip_install:
        print("[deps] Skipping dependency check (requested).")
        return

    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find requirements file at {REQUIREMENTS_FILE}."
        )

    print(f"[deps] Ensuring dependencies from {REQUIREMENTS_FILE} are installed...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)]
    if upgrade:
        cmd.append("--upgrade")
    subprocess.check_call(cmd, cwd=ROOT_DIR)  # noqa: S603,S607 - controlled input
    print("[deps] Dependencies are ready.")


def ensure_chroma_dir(reset: bool = False) -> None:
    """Ensure the ChromaDB persist directory exists.
    
    Args:
        reset: If True, delete existing vector store data and start fresh.
    """
    if reset and CHROMA_DIR.exists():
        print(f"[chroma] Resetting vector store at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)
    
    if not CHROMA_DIR.exists():
        print(f"[chroma] Creating vector store directory at {CHROMA_DIR}...")
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[chroma] Vector store directory exists at {CHROMA_DIR}")


def check_env_vars() -> list[str]:
    """Check for required environment variables and return list of missing ones."""
    required = ["GOOGLE_API_KEY", "TAVILY_API_KEY"]
    optional = ["GNEWS_API_KEY", "NEWS_RAG_MODEL_NAME"]
    
    missing = [var for var in required if not os.environ.get(var)]
    
    if missing:
        print(f"[env] WARNING: Missing required environment variables: {', '.join(missing)}")
        print("[env] Please set these in your .env file or environment.")
    
    # Check optional vars
    for var in optional:
        if not os.environ.get(var):
            print(f"[env] Note: Optional variable {var} not set.")
    
    return missing


def start_process(label: str, command: Sequence[str], env: dict[str, str]) -> subprocess.Popen:
    """Launch a child process and return the handle."""

    print(f"[{label}] {' '.join(command)}")
    return subprocess.Popen(  # noqa: S603 - command constructed above
        command,
        cwd=ROOT_DIR,
        env=env,
    )


def wait_for_backend(base_url: str, timeout: float) -> None:
    """Poll the backend health endpoint until it responds or timeout occurs."""

    health_url = f"{base_url.rstrip('/')}" + "/health"
    deadline = time.time() + timeout
    print(f"[backend] Waiting for health check at {health_url} ...")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3):  # noqa: S310
                print("[backend] Health check succeeded.")
                return
        except urllib.error.URLError:
            time.sleep(1.0)
    print(
        "[backend] Health check timed out. The frontend may fail to connect if the "
        "backend is still starting."
    )


def shutdown_process(proc: subprocess.Popen | None, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return

    print(f"[{label}] Stopping...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"[{label}] Terminate timed out. Killing...")
        proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Python dependencies are installed, initialize vector store, "
            "then start the FastAPI backend and Streamlit frontend together."
        )
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip running 'pip install -r requirements.txt' before launching.",
    )
    parser.add_argument(
        "--upgrade-deps",
        action="store_true",
        help="Upgrade all dependencies to latest versions.",
    )
    parser.add_argument(
        "--backend-host",
        default="127.0.0.1",
        help="Host/interface for the FastAPI server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="Port for the FastAPI server (default: 8000).",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8501,
        help="Port for the Streamlit app (default: 8501).",
    )
    parser.add_argument(
        "--backend-startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the backend health endpoint before continuing.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable uvicorn auto-reload (enabled by default).",
    )
    parser.add_argument(
        "--reset-vector-store",
        action="store_true",
        help="Delete existing vector store data and start fresh.",
    )
    parser.add_argument(
        "--legacy-mode",
        action="store_true",
        help="Use legacy summarize API instead of RAG API in the UI.",
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip checking for required environment variables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv
        env_file = ROOT_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            print(f"[env] Loaded environment from {env_file}")
    except ImportError:
        pass  # dotenv not installed, rely on system env vars

    # Check environment variables
    if not args.skip_env_check:
        missing = check_env_vars()
        if missing:
            print("[env] Continuing anyway, but some features may not work.")

    # Install/upgrade dependencies
    try:
        ensure_dependencies(skip_install=args.skip_install, upgrade=args.upgrade_deps)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[deps] Dependency installation failed: {exc}")
        return 1

    # Initialize vector store directory
    ensure_chroma_dir(reset=args.reset_vector_store)

    backend_base_url = f"http://{args.backend_host}:{args.backend_port}"

    env = os.environ.copy()
    env.setdefault("NEWS_RAG_API_BASE_URL", backend_base_url)
    
    # Set RAG mode based on --legacy-mode flag
    if args.legacy_mode:
        env["USE_RAG_API"] = "false"
        print("[mode] Running in LEGACY mode (summary-only, no vector retrieval)")
    else:
        env["USE_RAG_API"] = "true"
        print("[mode] Running in RAG mode (full retrieval-augmented generation)")
    
    # Set ChromaDB persist directory
    env.setdefault("CHROMA_PERSIST_DIR", str(CHROMA_DIR))

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        UVICORN_APP,
        "--host",
        args.backend_host,
        "--port",
        str(args.backend_port),
    ]
    if not args.no_reload:
        backend_cmd.append("--reload")

    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.port",
        str(args.frontend_port),
    ]

    backend_proc = frontend_proc = None
    try:
        backend_proc = start_process("backend", backend_cmd, env)
        wait_for_backend(backend_base_url, args.backend_startup_timeout)
        frontend_proc = start_process("frontend", frontend_cmd, env)

        print("[runner] Both services are running. Press Ctrl+C to stop.")
        print(f"[runner] Backend API: {backend_base_url}")
        print(f"[runner] Frontend UI: http://localhost:{args.frontend_port}")
        print(f"[runner] API Docs: {backend_base_url}/docs")
        
        while True:
            backend_status = backend_proc.poll() if backend_proc else 0
            frontend_status = frontend_proc.poll() if frontend_proc else 0

            if backend_status is not None:
                print(f"[backend] exited with status {backend_status}.")
                break
            if frontend_status is not None:
                print(f"[frontend] exited with status {frontend_status}.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[runner] Caught KeyboardInterrupt. Shutting down...")
    finally:
        shutdown_process(frontend_proc, "frontend")
        shutdown_process(backend_proc, "backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
