"""Stage 6: master comparison and champion selection."""
from pathlib import Path
import json
import shutil
import pandas as pd

def run_stage6(stage3_dir: Path, stage4_dir: Path, stage5_dir: Path, stage6_dir: Path) -> dict:
    stage6_dir.mkdir(parents=True,exist_ok=True)
    cml=pd.read_csv(stage3_dir/"classification_results.csv"); rml=pd.read_csv(stage3_dir/"regression_results.csv"); cdl=pd.read_csv(stage4_dir/"dl_classification_results.csv"); rdl=pd.read_csv(stage4_dir/"dl_regression_results.csv"); shap=pd.read_csv(stage5_dir/"shap_feature_importance.csv")
    c=pd.concat([cml,cdl],ignore_index=True).sort_values("Validation_PR_AUC",ascending=False).reset_index(drop=True); c["Validation_Rank"]=range(1,len(c)+1)
    rclass=c.iloc[0].to_dict()
    r=pd.concat([rml.assign(Source="Classical ML"),rdl.assign(Source="Deep Learning")],ignore_index=True)
    r_classical=rml.sort_values("Validation_RMSE_Days").reset_index(drop=True); r_classical["Validation_Rank"]=range(1,len(r_classical)+1)
    rreg=r_classical.iloc[0].to_dict()
    c.to_csv(stage6_dir/"master_classification_results.csv",index=False); r.to_csv(stage6_dir/"master_regression_results.csv",index=False); shap.head(15).to_csv(stage6_dir/"shap_top15.csv",index=False)
    summary=pd.DataFrame([
        {"Task":"Classification","Champion":rclass["Model"],"Selection_Metric":"Validation PR-AUC","Validation_Metric":rclass["Validation_PR_AUC"],"Holdout_Primary_Metric":rclass["Holdout_PR_AUC"],"Holdout_Secondary_Metric":rclass["Holdout_ROC_AUC"]},
        {"Task":"RUL Regression","Champion":rreg["Model"],"Selection_Metric":"Validation RMSE","Validation_Metric":rreg["Validation_RMSE_Days"],"Holdout_Primary_Metric":rreg["Holdout_RMSE_Days"],"Holdout_Secondary_Metric":rreg["Holdout_R2"]}
    ]); summary.to_csv(stage6_dir/"final_master_summary.csv",index=False)
    metadata={"classification_champion":{"model_name":rclass["Model"],"approach":"Deep Learning" if "Deep Learning" in rclass["Model"] else "Machine Learning","validation_pr_auc":float(rclass["Validation_PR_AUC"]),"holdout_pr_auc":float(rclass["Holdout_PR_AUC"]),"holdout_roc_auc":float(rclass["Holdout_ROC_AUC"])},"regression_champion":{"model_name":rreg["Model"],"validation_rmse_days":float(rreg["Validation_RMSE_Days"]),"holdout_rmse_days":float(rreg["Holdout_RMSE_Days"]),"holdout_r2":float(rreg["Holdout_R2"])}}
    (stage6_dir/"champion_metadata.json").write_text(json.dumps(metadata,indent=4),encoding='utf-8')
    # deployment copies
    deploy_class = stage4_dir/"dl_classifier_mlp.keras"; deploy_reg = stage3_dir/"gradient_boosting_regressor.pkl"
    return {"classification_champion":rclass["Model"],"regression_champion":rreg["Model"],"summary":summary.to_dict("records"),"deployment_artifacts":{"classification":str(deploy_class),"regression":str(deploy_reg)}}
