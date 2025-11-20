from __future__ import annotations

import subprocess
import sys
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_process(command: list[str], cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=str(cwd))


def main() -> int:
    processes: list[subprocess.Popen] = []
    try:
        print("Starting FastAPI backend (uvicorn)...")
        processes.append(
            run_process(
                ["uvicorn", "api.main:app", "--reload", "--port", "8000"],
                ROOT,
            )
        )

        print("Starting Next.js frontend...")
        processes.append(run_process(["npm", "run", "dev"], ROOT / "web"))

        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down processes...")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
