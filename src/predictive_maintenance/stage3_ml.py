"""Stage 3: classical ML."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error, r2_score
from .features import FEATURE_COLUMNS, CLASSIFICATION_TARGET, REGRESSION_TARGET
from .io import save_csv

def _classifier_models(seed=42):
    return {
        "Logistic Regression":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=seed))]),
        "Random Forest Classifier":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=seed,n_jobs=-1))]),
        "Gradient Boosting Classifier":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",GradientBoostingClassifier(n_estimators=100,random_state=seed))])}

def _regressor_models(seed=42):
    return {
        "Random Forest Regressor":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",RandomForestRegressor(n_estimators=100,max_depth=15,random_state=seed,n_jobs=-1))]),
        "Gradient Boosting Regressor":Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("model",GradientBoostingRegressor(n_estimators=100,max_depth=5,random_state=seed))])}

def run_stage3(stage2_dir: Path, stage3_dir: Path, seed=42) -> dict:
    stage3_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(stage2_dir/"train_raw_partition.csv"); val=pd.read_csv(stage2_dir/"validation_raw_partition.csv"); test=pd.read_csv(stage2_dir/"test_raw_partition.csv")
    with open(stage2_dir/"feature_contract.json",encoding="utf-8") as f: contract=json.load(f)
    features=contract["numerical_features"]; cls=contract["classification_target"]; rul=contract["regression_target"]

    Xtr=train[features]; Xv=val[features]; Xte=test[features]; ytr=train[cls].astype(int); yv=val[cls].astype(int); yte=test[cls].astype(int)
    class_rows=[]
    for name,model in _classifier_models(seed).items():
        model.fit(Xtr,ytr); vp=model.predict_proba(Xv)[:,1]; tp=model.predict_proba(Xte)[:,1]
        class_rows.append({"Model":name,"Task":"Classification","Validation_PR_AUC":average_precision_score(yv,vp),"Validation_ROC_AUC":roc_auc_score(yv,vp),"Holdout_PR_AUC":average_precision_score(yte,tp),"Holdout_ROC_AUC":roc_auc_score(yte,tp)})
        joblib.dump(model,stage3_dir/(name.lower().replace(' ','_').replace('-','')+'.pkl'))
    cdf=pd.DataFrame(class_rows).sort_values("Validation_PR_AUC",ascending=False); save_csv(cdf,stage3_dir/"classification_results.csv")

    tr=train[train[rul].notna()]; vr=val[val[rul].notna()]; ter=test[test[rul].notna()]
    Xtr=tr[features]; Xv=vr[features]; Xte=ter[features]; ytr=tr[rul].astype(float); yv=vr[rul].astype(float); yte=ter[rul].astype(float)
    reg_rows=[]
    for name,model in _regressor_models(seed).items():
        model.fit(Xtr,ytr); vp=model.predict(Xv); tp=model.predict(Xte)
        reg_rows.append({"Model":name,"Task":"Regression","Validation_RMSE_Days":np.sqrt(mean_squared_error(yv,vp)),"Validation_R2":r2_score(yv,vp),"Holdout_RMSE_Days":np.sqrt(mean_squared_error(yte,tp)),"Holdout_R2":r2_score(yte,tp)})
        joblib.dump(model,stage3_dir/(name.lower().replace(' ','_').replace('-','')+'.pkl'))
    rdf=pd.DataFrame(reg_rows).sort_values("Validation_RMSE_Days",ascending=True); save_csv(rdf,stage3_dir/"regression_results.csv")
    return {"classification_results":cdf.to_dict("records"),"regression_results":rdf.to_dict("records"),"feature_count":len(features)}
