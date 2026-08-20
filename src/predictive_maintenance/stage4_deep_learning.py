"""Stage 4: memory-safe deep learning."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_error, r2_score

def run_stage4(stage2_dir: Path, stage4_dir: Path, epochs=8, batch_size=512, sequence_length=12):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.metrics import AUC
    tf.keras.backend.set_floatx('float32')
    stage4_dir.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(stage2_dir/"train_raw_partition.csv"); val=pd.read_csv(stage2_dir/"validation_raw_partition.csv"); test=pd.read_csv(stage2_dir/"test_raw_partition.csv")
    with open(stage2_dir/"feature_contract.json",encoding="utf-8") as f: contract=json.load(f)
    features=contract["numerical_features"]; cls=contract["classification_target"]; rul=contract["regression_target"]
    Xtr=train[features].to_numpy(np.float32); Xv=val[features].to_numpy(np.float32); Xte=test[features].to_numpy(np.float32)
    imp=SimpleImputer(strategy='median'); Xtr=imp.fit_transform(Xtr).astype(np.float32); Xv=imp.transform(Xv).astype(np.float32); Xte=imp.transform(Xte).astype(np.float32)
    scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr).astype(np.float32); Xv=scaler.transform(Xv).astype(np.float32); Xte=scaler.transform(Xte).astype(np.float32)
    joblib.dump(imp,stage4_dir/"dl_feature_imputer.pkl"); joblib.dump(scaler,stage4_dir/"dl_feature_scaler.pkl")

    ytr=train[cls].to_numpy(np.float32); yv=val[cls].to_numpy(np.float32); yte=test[cls].to_numpy(np.float32)
    mlp=Sequential([Input(shape=(len(features),)),Dense(64,activation='relu'),Dropout(.3),Dense(32,activation='relu'),Dropout(.2),Dense(1,activation='sigmoid')])
    mlp.compile(optimizer='adam',loss='binary_crossentropy',metrics=[AUC(name='roc_auc',curve='ROC'),AUC(name='pr_auc',curve='PR')])
    mlp.fit(Xtr,ytr,validation_data=(Xv,yv),epochs=epochs,batch_size=batch_size,callbacks=[EarlyStopping(monitor='val_pr_auc',mode='max',patience=2,restore_best_weights=True)],verbose=1)
    vp=mlp.predict(Xv,batch_size=4096,verbose=0).flatten(); tp=mlp.predict(Xte,batch_size=4096,verbose=0).flatten()
    cdf=pd.DataFrame([{"Model":"Deep Learning MLP","Task":"Classification","Validation_PR_AUC":average_precision_score(yv,vp),"Validation_ROC_AUC":roc_auc_score(yv,vp),"Holdout_PR_AUC":average_precision_score(yte,tp),"Holdout_ROC_AUC":roc_auc_score(yte,tp)}]); cdf.to_csv(stage4_dir/"dl_classification_results.csv",index=False); mlp.save(stage4_dir/"dl_classifier_mlp.keras")

    tr=train[train[rul].notna()]; vr=val[val[rul].notna()]; ter=test[test[rul].notna()]
    ytr_r=tr[rul].to_numpy(np.float32); yv_r=vr[rul].to_numpy(np.float32); yte_r=ter[rul].to_numpy(np.float32)
    # Align feature matrices to the valid RUL rows.
    # Build from original row indexes to preserve chronology.
    Xtr_r=Xtr[tr.index.to_numpy()-train.index.to_numpy()[0]] if len(tr) else np.empty((0,len(features)),np.float32)
    Xv_r=Xv[vr.index.to_numpy()-val.index.to_numpy()[0]] if len(vr) else np.empty((0,len(features)),np.float32)
    Xte_r=Xte[ter.index.to_numpy()-test.index.to_numpy()[0]] if len(ter) else np.empty((0,len(features)),np.float32)
    target_scaler=StandardScaler(); ytr_s=target_scaler.fit_transform(ytr_r.reshape(-1,1)).astype(np.float32).ravel(); yv_s=target_scaler.transform(yv_r.reshape(-1,1)).astype(np.float32).ravel(); _=target_scaler.transform(yte_r.reshape(-1,1))
    joblib.dump(target_scaler,stage4_dir/"dl_rul_target_scaler.pkl")
    def ds(X,y): return tf.keras.utils.timeseries_dataset_from_array(X,y,sequence_length=sequence_length,sequence_stride=1,sampling_rate=1,batch_size=batch_size,shuffle=False)
    train_ds=ds(Xtr_r,ytr_s); val_ds=ds(Xv_r,yv_s); test_ds=ds(Xte_r,yte_r)  # targets are raw; scaler only applied to training/validation for fitting
    # Recreate datasets with scaled targets for fitting.
    train_ds=ds(Xtr_r,ytr_s); val_ds=ds(Xv_r,yv_s)
    lstm=Sequential([Input(shape=(sequence_length,len(features))),LSTM(64,activation='tanh',return_sequences=True),Dropout(.2),LSTM(32,activation='tanh'),Dropout(.2),Dense(1)])
    lstm.compile(optimizer='adam',loss='mse',metrics=['mae'])
    lstm.fit(train_ds,validation_data=val_ds,epochs=epochs,callbacks=[EarlyStopping(monitor='val_loss',mode='min',patience=2,restore_best_weights=True)],verbose=1)
    pred_s=lstm.predict(ds(Xte_r,target_scaler.transform(yte_r.reshape(-1,1)).astype(np.float32).ravel()),verbose=0).flatten()
    y_seq=yte_r[sequence_length-1:]; pred_s=pred_s[:len(y_seq)]; pred=target_scaler.inverse_transform(pred_s.reshape(-1,1)).ravel()
    rdf=pd.DataFrame([{"Model":"LSTM Regressor","Task":"Regression","Holdout_RMSE_Days":np.sqrt(mean_squared_error(y_seq,pred)),"Holdout_R2":r2_score(y_seq,pred),"Sequence_Length":sequence_length,"History_Hours":sequence_length*0.5}]); rdf.to_csv(stage4_dir/"dl_regression_results.csv",index=False); lstm.save(stage4_dir/"dl_regressor_lstm.keras")
    metadata={"feature_count":len(features),"sequence_length":sequence_length,"history_hours":sequence_length*.5,"preprocessing":"train-only median imputation and StandardScaler","dtype":"float32","models":["Deep Learning MLP","LSTM Regressor"]}
    (stage4_dir/"stage4_deep_learning_metadata.json").write_text(json.dumps(metadata,indent=4),encoding='utf-8')
    return {"classification_results":cdf.to_dict("records"),"regression_results":rdf.to_dict("records")}
