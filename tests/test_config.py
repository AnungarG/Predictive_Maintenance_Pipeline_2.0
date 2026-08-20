import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from predictive_maintenance.config import load_config

def test_config():
    cfg=load_config()
    assert cfg["pipeline_name"]=="Predictive Maintenance Pipeline 2.0"
