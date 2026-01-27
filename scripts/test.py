import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/"],
        cwd=ROOT,
        check=True,
    )
