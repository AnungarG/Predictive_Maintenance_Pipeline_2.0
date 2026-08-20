"""Stage 1: raw audit and EDA outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from .features import CLASSIFICATION_TARGET, REGRESSION_TARGET, PHYSICAL_BOUNDS
from .io import load_csv, save_csv, save_json

def run_stage1(raw_path: Path, stage_dir: Path) -> dict:
    stage_dir.mkdir(parents=True, exist_ok=True)
    df = load_csv(raw_path)
    if "timestamp" not in df.columns:
        raise ValueError("Required column 'timestamp' is missing.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    schema = pd.DataFrame({"column": df.columns, "data_type": df.dtypes.astype(str).values})
    save_csv(schema, stage_dir/"schema_audit.csv")

    duplicates = pd.DataFrame({
        "audit_item": ["exact_duplicate_rows", "equipment_timestamp_duplicate_rows"],
        "count": [int(df.duplicated().sum()), int(df.duplicated(["equipment_id","timestamp"]).sum())]
    })
    save_csv(duplicates, stage_dir/"duplicate_audit.csv")

    ts = pd.DataFrame({
        "metric": ["invalid_or_missing_timestamps","minimum_timestamp","maximum_timestamp","unique_timestamps"],
        "value": [int(df["timestamp"].isna().sum()), df["timestamp"].min(), df["timestamp"].max(), int(df["timestamp"].nunique())]
    })
    save_csv(ts, stage_dir/"timestamp_audit.csv")

    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().mean()*100).values
    }).sort_values("missing_percentage", ascending=False)
    save_csv(missing, stage_dir/"missing_value_audit.csv")

    numeric = df.select_dtypes(include=[np.number])
    stats = numeric.describe().T
    stats["skewness"] = numeric.skew(numeric_only=True)
    save_csv(stats.reset_index(names="variable"), stage_dir/"raw_statistical_profile.csv")

    failure_summary = (df[CLASSIFICATION_TARGET].value_counts(dropna=False).rename_axis("class").reset_index(name="count"))
    failure_summary["percentage"] = failure_summary["count"] / len(df) * 100
    save_csv(failure_summary, stage_dir/"failure_within_24h_summary.csv")

    rul = df[REGRESSION_TARGET]
    rul_summary = pd.DataFrame({"metric":["missing","negative","mean","median","minimum","maximum"],
        "value":[rul.isna().sum(), (rul<0).sum(), rul.mean(), rul.median(), rul.min(), rul.max()]})
    save_csv(rul_summary, stage_dir/"rul_audit.csv")

    physical_rows=[]
    for col,(lower,upper) in PHYSICAL_BOUNDS.items():
        if col not in df.columns: continue
        mask=df[col].notna()
        if lower is not None: mask &= df[col] < lower
        if upper is not None: mask |= df[col].notna() & (df[col] > upper)
        invalid=int(mask.sum())
        physical_rows.append({"variable":col,"lower_bound":lower,"upper_bound":upper,"invalid_count":invalid,"invalid_pct":invalid/len(df)*100})
    save_csv(pd.DataFrame(physical_rows), stage_dir/"physical_validity_audit.csv")

    outlier_rows=[]
    for col in numeric.columns:
        s=numeric[col].dropna()
        if s.empty: continue
        q1,q3=s.quantile([0.25,0.75]); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
        count=int(((numeric[col]<lo)|(numeric[col]>hi)).sum())
        outlier_rows.append({"variable":col,"lower_iqr":lo,"upper_iqr":hi,"outlier_count":count,"outlier_pct":count/len(df)*100})
    outliers=pd.DataFrame(outlier_rows).sort_values("outlier_count",ascending=False)
    save_csv(outliers, stage_dir/"outlier_audit.csv")

    leakage_vars=[c for c in ["failure_within_72h","failure_within_7d","rul_days","rul_censored","failure_rate"] if c in df.columns]
    leakage=pd.DataFrame({"variable":leakage_vars,"present":[True]*len(leakage_vars)})
    save_csv(leakage, stage_dir/"leakage_screening.csv")

    # Target-status medians for the key condition variables used in Chapter 4.
    target=df[CLASSIFICATION_TARGET].eq(1)
    rows=[]
    for col in ["overall_vibration","vibration","oil_particles_ppm","oil_pressure","lubrication_health_index","load_factor","wear_level"]:
        if col in df.columns:
            normal=df.loc[~target,col].median(); failure=df.loc[target,col].median()
            rows.append({"sensor":col,"normal_n":int((~target & df[col].notna()).sum()),"failure_n":int((target & df[col].notna()).sum()),"normal_median":normal,"failure_median":failure,"median_difference":failure-normal})
    save_csv(pd.DataFrame(rows), stage_dir/"sensor_failure_comparison.csv")

    summary={"rows":len(df),"columns":df.shape[1],"exact_duplicates":int(df.duplicated().sum()),"equipment_timestamp_duplicates":int(df.duplicated(["equipment_id","timestamp"]).sum()),"missing_value_columns":int((df.isna().sum()>0).sum()),"equipment_assets":int(df["equipment_id"].nunique()) if "equipment_id" in df.columns else None,"trains":int(df["train"].nunique()) if "train" in df.columns else None}
    save_json(summary, stage_dir/"stage1_summary.json")
    return summary
