"""One-command orchestration for Predictive Maintenance Pipeline 2.0."""
from .config import load_config, project_root
from .stage1_audit import run_stage1
from .stage2_cleaning import run_stage2
from .stage3_ml import run_stage3
from .stage4_deep_learning import run_stage4
from .stage5_explainability import run_stage5
from .stage6_comparison import run_stage6
from pathlib import Path

def run_pipeline() -> dict:
    cfg=load_config(); root=project_root(); raw=root/cfg["paths"]["raw_data"]; out=root/cfg["paths"]["output_root"]
    s1=out/cfg["paths"]["stage1"]; s2=out/cfg["paths"]["stage2"]; s3=out/cfg["paths"]["stage3"]; s4=out/cfg["paths"]["stage4"]; s5=out/cfg["paths"]["stage5"]; s6=out/cfg["paths"]["stage6"]
    results={}
    results["stage1"]=run_stage1(raw,s1)
    results["stage2"]=run_stage2(raw,s2)
    results["stage3"]=run_stage3(s2,s3,cfg["validation"]["random_seed"])
    results["stage4"]=run_stage4(s2,s4,cfg["deep_learning"]["epochs"],cfg["deep_learning"]["batch_size"],cfg["deep_learning"]["sequence_length"])
    results["stage5"]=run_stage5(s2,s3,s5,cfg["shap"]["sample_size"],cfg["validation"]["random_seed"],cfg["shap"]["top_n"])
    results["stage6"]=run_stage6(s3,s4,s5,s6)
    return results
