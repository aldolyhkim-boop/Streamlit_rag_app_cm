"""Embed the standalone AI Dashboard Builder in an isolated Streamlit process."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

import streamlit as st

BUILDER_ROOT = Path(__file__).resolve().parents[1] / "ai-dashboard-builder-main"
BUILDER_DATA = BUILDER_ROOT
BUILDER_LOG = Path(__file__).resolve().parents[1] / ".ai_dashboard_builder" / "streamlit.log"
# Keep the embedded app on the same hostname as the parent app so browser
# cookie and CSRF checks continue to work inside the iframe.
BUILDER_HOST = "localhost"
BUILDER_PORT = 8503
BUILDER_URL = f"http://{BUILDER_HOST}:{BUILDER_PORT}"


def _port_is_open() -> bool:
    try:
        with socket.create_connection((BUILDER_HOST, BUILDER_PORT), timeout=0.2):
            return True
    except OSError:
        return False


def _server_is_healthy() -> bool:
    try:
        with urlopen(f"{BUILDER_URL}/_stcore/health", timeout=1) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


@st.cache_resource(show_spinner=False)
def _start_builder_server() -> subprocess.Popen | None:
    if not (BUILDER_ROOT / "app.py").is_file():
        raise FileNotFoundError(f"Standalone AI Dashboard Builder was not found at {BUILDER_ROOT}.")
    if _port_is_open():
        if _server_is_healthy():
            return None
        raise RuntimeError(
            f"Port {BUILDER_PORT} is already in use. Close the other app using that port and try again."
        )

    BUILDER_LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("DASHBOARD_DATA_DIR", str(BUILDER_DATA))
    log_file = BUILDER_LOG.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address",
                BUILDER_HOST,
                "--server.port",
                str(BUILDER_PORT),
                "--server.headless",
                "true",
            ],
            cwd=BUILDER_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        log_file.close()
    return process


def _wait_for_builder_server(process: subprocess.Popen | None, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            details = BUILDER_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"The Builder server stopped during startup.\n\n{details}")
        if _server_is_healthy():
            return
        time.sleep(0.25)
    raise TimeoutError(f"The Builder server did not start within {timeout:g} seconds.")


def render_ai_dashboard_builder() -> None:
    """Render the original Builder app without sharing its Python or UI state with Smart EDA."""
    try:
        with st.spinner("Starting AI Dashboard Builder..."):
            process = _start_builder_server()
            _wait_for_builder_server(process)
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError) as exc:
        st.error(f"Could not start AI Dashboard Builder: {exc}")
        return

    st.iframe(BUILDER_URL, height=1100)
