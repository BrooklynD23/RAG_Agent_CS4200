#!/usr/bin/env python3
"""Utility to verify dependencies and launch backend + frontend together."""

from __future__ import annotations

import argparse
import os
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


def ensure_dependencies(skip_install: bool) -> None:
    """Ensure Python dependencies defined in requirements.txt are installed."""

    if skip_install:
        print("[deps] Skipping dependency check (requested).")
        return

    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find requirements file at {REQUIREMENTS_FILE}."
        )

    print(f"[deps] Ensuring dependencies from {REQUIREMENTS_FILE} are installed...")
    subprocess.check_call(  # noqa: S603,S607 - controlled input
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=ROOT_DIR,
    )
    print("[deps] Dependencies are ready.")


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
            "Verify Python dependencies are installed, then start the FastAPI backend "
            "and Streamlit frontend together."
        )
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip running 'pip install -r requirements.txt' before launching.",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        ensure_dependencies(skip_install=args.skip_install)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[deps] Dependency installation failed: {exc}")
        return 1

    backend_base_url = f"http://{args.backend_host}:{args.backend_port}"

    env = os.environ.copy()
    env.setdefault("NEWS_RAG_API_BASE_URL", backend_base_url)

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
