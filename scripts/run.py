import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    subprocess.run(["docker", "compose", "up"], cwd=ROOT, check=True)
