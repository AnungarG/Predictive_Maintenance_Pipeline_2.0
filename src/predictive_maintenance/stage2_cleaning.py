"""Stage 2: cleaning, leakage control and chronological partition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .features import FEATURE_COLUMNS, CLASSIFICATION_TARGET, REGRESSION_TARGET, EXCLUDED_FEATURES, PHYSICAL_BOUNDS
from .io import load_csv, save_csv, save_json

def run_stage2(raw_path: Path, stage_dir: Path) -> dict:
    stage_dir.mkdir(parents=True, exist_ok=True)
    df=load_csv(raw_path)
    df["timestamp"]=pd.to_datetime(df["timestamp"], errors="coerce")
    original_rows=len(df)
    df=df.dropna(subset=["timestamp"]).copy()
    exact=int(df.duplicated().sum()); df=df.drop_duplicates(keep="first").copy()
    df=df.sort_values(["equipment_id","timestamp","row_id"], kind="mergesort").reset_index(drop=True)
    equip_dup=int(df.duplicated(["equipment_id","timestamp"], keep=False).sum())
    groups=df.groupby(["equipment_id","timestamp"],dropna=False).size().reset_index(name="records_at_same_timestamp")
    save_csv(groups[groups["records_at_same_timestamp"]>1],stage_dir/"equipment_timestamp_duplicate_groups.csv")
    df=df.drop_duplicates(["equipment_id","timestamp"],keep="first").reset_index(drop=True)

    cleaning=[]
    for col,(lower,upper) in PHYSICAL_BOUNDS.items():
        if col not in df.columns: continue
        mask=df[col].notna()
        if lower is not None: mask &= df[col] < lower
        if upper is not None: mask |= df[col].notna() & (df[col] > upper)
        count=int(mask.sum())
        if count: df.loc[mask,col]=np.nan
        cleaning.append({"variable":col,"lower_bound":lower,"upper_bound":upper,"converted_to_nan":count})
    save_csv(pd.DataFrame(cleaning),stage_dir/"physical_cleaning_log.csv")

    actual_excluded=[c for c in EXCLUDED_FEATURES if c in df.columns]
    actual_features=[c for c in FEATURE_COLUMNS if c in df.columns]
    if actual_features != FEATURE_COLUMNS:
        missing=[c for c in FEATURE_COLUMNS if c not in df.columns]
        raise ValueError(f"Feature contract mismatch. Missing features: {missing}")

    unique_times=np.sort(df["timestamp"].dropna().unique())
    train_idx=int(len(unique_times)*0.64); val_idx=int(len(unique_times)*0.80)
    train_end=unique_times[train_idx-1]; validation_end=unique_times[val_idx-1]
    df["data_partition"]=np.select([df["timestamp"]<=train_end, df["timestamp"]<=validation_end],["train","validation"],default="test")
    train=df[df["data_partition"]=="train"].copy(); val=df[df["data_partition"]=="validation"].copy(); test=df[df["data_partition"]=="test"].copy()
    if not (train["timestamp"].max()<val["timestamp"].min()<test["timestamp"].min()): raise AssertionError("Temporal partition audit failed.")

    feature_contract={"classification_target":CLASSIFICATION_TARGET,"regression_target":REGRESSION_TARGET,"excluded_features":actual_excluded,"numerical_features":FEATURE_COLUMNS,"failure_rate_status":"EXCLUDED_PENDING_PROVENANCE_REVIEW"}
    save_json(feature_contract,stage_dir/"feature_contract.json")
    save_csv(pd.DataFrame({"partition":["train","validation","test"],"rows":[len(train),len(val),len(test)],"percentage":[len(train)/len(df)*100,len(val)/len(df)*100,len(test)/len(df)*100],"start_time":[train.timestamp.min(),val.timestamp.min(),test.timestamp.min()],"end_time":[train.timestamp.max(),val.timestamp.max(),test.timestamp.max()]}),stage_dir/"partition_summary.csv")
    df.to_csv(stage_dir/"NLNG_cleaned_leakage_controlled.csv",index=False)
    train.to_csv(stage_dir/"train_raw_partition.csv",index=False); val.to_csv(stage_dir/"validation_raw_partition.csv",index=False); test.to_csv(stage_dir/"test_raw_partition.csv",index=False)
    save_json({"strategy":"chronological_timestamp_split","train_percentage_target":64,"validation_percentage_target":16,"test_percentage_target":20,"train_end":str(train_end),"validation_end":str(validation_end),"test_start":str(test.timestamp.min()),"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test)},stage_dir/"temporal_split_metadata.json")
    return {"original_rows":original_rows,"exact_duplicates_removed":exact,"equipment_timestamp_duplicate_rows":equip_dup,"rows_after_dedup":len(df),"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),"feature_count":len(FEATURE_COLUMNS)}
