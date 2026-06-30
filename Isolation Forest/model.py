import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# CONFIG
data_atual = "2026-06-20"
CSV_PATH = f"Analise_{data_atual}.csv"
FREQ = "1min"                 
ROLL_15 = 15                  
ROLL_60 = 60                  
CONTAMINATION = 0.005         
RANDOM_STATE = 42

# LOAD
usecols = ["timestamp", "host_name", "Ping", "Packet loss", "Response time"]
df = pd.read_csv(CSV_PATH, usecols=usecols)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
for col in ["Ping", "Packet loss", "Response time"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["timestamp", "host_name"]).sort_values(["host_name", "timestamp"])

# RESAMPLE
def resample_host(g):
    g = g.set_index("timestamp")
    out = g.resample(FREQ).agg({
        "Ping": "min",            
        "Packet loss": "mean",
        "Response time": "mean"
    })
    out["host_name"] = g["host_name"].iloc[0]
    return out.reset_index()

df_1m = df.groupby("host_name", group_keys=False).apply(resample_host)
df_1m["is_down"] = (df_1m["Ping"] == 0).astype(int)

# DEGRADATION DETECTION & FEATURE ENGINEERING
feature_cols = [
    "rt", "pl", "rt_delta", "pl_delta",
    "rt_roll_mean_15", "rt_iqr_15",       
    "pl_roll_mean_15", "pl_roll_max_15", "pl_iqr_15", 
    "pl_nonzero_rate_60", "outages_24h", "micro_outages_60m"    
]

anomaly_rows = []

for host, g in df_1m.groupby("host_name"):
    g = g.sort_values("timestamp").copy()
    
    # Contagem de Outages
    g["outage_start"] = ((g["is_down"] == 1) & (g["is_down"].shift(1) == 0)).astype(int)
    g["outages_24h"] = g["outage_start"].rolling(1440, min_periods=1).sum()
    g["micro_outages_60m"] = g["outage_start"].rolling(60, min_periods=1).sum()
    
    g_up = g[g["Ping"] == 1].copy()
    if len(g_up) < 300:
        continue
    
    g_up["rt"] = g_up["Response time"]
    g_up["pl"] = g_up["Packet loss"]
    g_up["rt_delta"] = g_up["rt"].diff()
    g_up["pl_delta"] = g_up["pl"].diff()
    
    # Jitter e dispersao via Interquartile Range (IQR)
    g_up["rt_roll_mean_15"] = g_up["rt"].rolling(ROLL_15, min_periods=5).mean()
    rt_q75 = g_up["rt"].rolling(ROLL_15, min_periods=5).quantile(0.75)
    rt_q25 = g_up["rt"].rolling(ROLL_15, min_periods=5).quantile(0.25)
    g_up["rt_iqr_15"] = rt_q75 - rt_q25
    
    g_up["pl_roll_mean_15"] = g_up["pl"].rolling(ROLL_15, min_periods=5).mean()
    g_up["pl_roll_max_15"]  = g_up["pl"].rolling(ROLL_15, min_periods=5).max()
    pl_q75 = g_up["pl"].rolling(ROLL_15, min_periods=5).quantile(0.75)
    pl_q25 = g_up["pl"].rolling(ROLL_15, min_periods=5).quantile(0.25)
    g_up["pl_iqr_15"] = pl_q75 - pl_q25
    
    g_up["pl_nonzero_rate_60"] = (g_up["pl"] > 0).rolling(ROLL_60, min_periods=10).mean()
    
    X = g_up[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < 200:
        continue
    
    # Robust Scaling
    X_med = X.median()
    X_iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0)
    Xs = (X - X_med) / X_iqr
    
    # Isolation Forest Model
    model = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(Xs)
    
    scores = -model.score_samples(Xs)   
    preds = model.predict(Xs)           
    
    out = g_up.loc[X.index, ["timestamp", "host_name", "Ping", "Packet loss", "Response time"]].copy()
    out["anomaly"] = (preds == -1).astype(int)
    
    # Penalidade por Packet Loss e Threshold Critico SCADA
    out["score"] = scores + (out["Packet loss"] / 100.0) * 0.5
    out.loc[out["Packet loss"] >= 15.0, "anomaly"] = 1
    
    out = out[out["anomaly"] == 1].sort_values("score", ascending=False)
    anomaly_rows.append(out)

degradation = pd.concat(anomaly_rows, ignore_index=True) if anomaly_rows else pd.DataFrame()
degradation.to_csv(f"degradation_{data_atual}.csv", index=False)
print("Finished")