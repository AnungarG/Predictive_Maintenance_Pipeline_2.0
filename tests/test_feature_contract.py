import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from predictive_maintenance.features import FEATURE_COLUMNS

def test_feature_contract():
    assert len(FEATURE_COLUMNS)==21
    assert len(set(FEATURE_COLUMNS))==21
