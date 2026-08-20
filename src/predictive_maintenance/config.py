from pathlib import Path
import json

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_config() -> dict:
    path = project_root() / "configs" / "pipeline_config.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def output_dir(stage_key: str) -> Path:
    cfg = load_config()
    root = project_root() / cfg["paths"]["output_root"]
    return root / cfg["paths"][stage_key]

