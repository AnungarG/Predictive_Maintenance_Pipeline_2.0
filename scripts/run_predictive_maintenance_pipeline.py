import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "src"
if str(src) not in sys.path: sys.path.insert(0,str(src))
from predictive_maintenance.runner import run_pipeline

if __name__ == "__main__":
    run_pipeline()
