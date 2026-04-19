# root/start_hal_services.py
import asyncio
import subprocess
import websockets
import sys
import os
import socket
import time
import signal
import psycopg2
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Paths
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = BASE_DIR
env = os.environ.copy()
env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

VOSK_SCRIPT = os.path.join(ROOT, "servers", "vosk_server.py")
LLM_SCRIPT  = os.path.join(ROOT, "servers", "llm_server.py")
GATEWAY_SCRIPT = os.path.join(ROOT, "gateway", "hal_server_gateway.py")

LOG_DIR = os.path.join(ROOT, "logging")
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------
# Process Management
# -------------------------

def wait_for_port(host, port, timeout=60):
    start = time.time()
    print(f"[Launcher] wait_for_port: host={host} port={port} timeout={timeout}")
    while True:
        try:
            with socket.create_connection((host, port), timeout=5):
                print(f"[Launcher] Port {host}:{port} is open.")
                return True
        except OSError as e:
            print(f"[Launcher] Port {host}:{port} not ready yet: {e!r}")

        if time.time() - start > timeout:
            raise TimeoutError(f"Port {host}:{port} did not open in time.")

        time.sleep(0.2)


def start_process(name, script_path, log_name):
    log_path = os.path.join(LOG_DIR, log_name)
    log_file = open(log_path, "a", buffering=1)

    print(f"[Launcher] Starting {name} -> {script_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
        cwd=ROOT,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )

    return process, log_file

def stop_processes(processes):
    print("[Launcher] Shutting down services...")

    for proc, log in processes:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
        except Exception:
            pass
        log.close()

    print("[Launcher] All services stopped.")

def check_sql():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            connect_timeout=3
        )
        conn.close()
        print("[Launcher] SQL connection OK.")
    except Exception as e:
        print(f"[Launcher] SQL connection FAILED: {e}")
        raise

# -------------------------
# Launch All Services
# -------------------------

async def launch_all_services():
    processes = []

    print("[Launcher] Checking SQL...")
    check_sql()

    # Start Vosk server
    vosk_proc, vosk_log = start_process(
        "Vosk Server",
        VOSK_SCRIPT,
        "vosk.log"
    )
    processes.append((vosk_proc, vosk_log))
    await asyncio.to_thread(
        wait_for_port,
        os.getenv("VOSK_CONNECT_HOST", "127.0.0.1"),
        int(os.getenv("VOSK_CONNECT_PORT", 2700))
    )
    print("[Launcher] Vosk server ready.")

    # Start LLM server
    llm_proc, llm_log = start_process(
        "LLM Server",
        LLM_SCRIPT,
        "llm.log"
    )
    processes.append((llm_proc, llm_log))
    await asyncio.to_thread(
        wait_for_port,
        os.getenv("LLM_CONNECT_HOST", "127.0.0.1"),
        int(os.getenv("LLM_CONNECT_PORT", 8765))
    )
    print("[Launcher] LLM server ready.")

    # Start Gateway
    gateway_proc, gateway_log = start_process(
        "HalServerGateway",
        GATEWAY_SCRIPT,
        "gateway.log"
    )
    processes.append((gateway_proc, gateway_log))
    await asyncio.to_thread(
        wait_for_port,
        os.getenv("GATEWAY_CONNECT_HOST", "127.0.0.1"),
        int(os.getenv("GATEWAY_CONNECT_PORT", 9000))
    )
    print("[Launcher] Gateway ready.")

    return processes

# -------------------------
# Main Entry Point
# -------------------------

async def async_main():
    print("[Launcher] Initializing Hal services...")

    processes = await launch_all_services()

    print("[Launcher] Press Ctrl+C to close gateway.")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        stop_processes(processes)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()