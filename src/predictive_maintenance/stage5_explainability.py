"""Stage 5: SHAP explainability for the Random Forest classifier."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

def run_stage5(stage2_dir: Path, stage3_dir: Path, stage5_dir: Path, sample_size=5000, seed=42, top_n=15):
    import shap
    stage5_dir.mkdir(parents=True,exist_ok=True)
    with open(stage2_dir/"feature_contract.json",encoding="utf-8") as f: contract=json.load(f)
    features=contract["numerical_features"]
    df=pd.read_csv(stage2_dir/"NLNG_cleaned_leakage_controlled.csv")
    X=df[features].sample(n=min(sample_size,len(df)),random_state=seed)
    pipeline=joblib.load(stage3_dir/"random_forest_classifier.pkl")
    imputer=pipeline.named_steps["imputer"]; model=pipeline.named_steps["model"]
    Xt=imputer.transform(X)
    names=list(imputer.get_feature_names_out(features))
    values=shap.TreeExplainer(model).shap_values(Xt)
    if isinstance(values,list): arr=values[1] if len(values)>1 else values[0]
    elif hasattr(values,'values'): arr=values.values
    else: arr=np.asarray(values)
    if arr.ndim==3: arr=arr[:,:,1]
    imp=pd.DataFrame({"Feature":names,"Mean_Absolute_SHAP":np.abs(arr).mean(axis=0)}).sort_values("Mean_Absolute_SHAP",ascending=False)
    imp.to_csv(stage5_dir/"shap_feature_importance.csv",index=False)
    top=imp.head(top_n)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6)); plt.barh(top["Feature"][::-1],top["Mean_Absolute_SHAP"][::-1]); plt.xlabel("Mean |SHAP Value|"); plt.title("Top Failure-Prediction Drivers"); plt.tight_layout(); plt.savefig(stage5_dir/"shap_feature_importance.png",dpi=300); plt.close()
    except Exception:
        pass
    meta={"model":"Random Forest Classifier","sample_size":len(X),"original_feature_count":len(features),"explained_feature_count":len(names)}
    (stage5_dir/"stage5_shap_metadata.json").write_text(json.dumps(meta,indent=4),encoding='utf-8')
    return imp.to_dict("records")
