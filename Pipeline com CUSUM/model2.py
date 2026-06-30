
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings


# ================
# CONFIGURACOES
# ================
CSV_PATH        = "Analise.csv"
OUT_HEALTH      = "health_index.csv"
OUT_OUTAGES     = "outages.csv"
OUT_DEGRADATION = "degradation.csv"
OUT_RANKING     = "ranking_criticidade.csv"

FREQ            = "1min"

# --- Isolation Forest ---
BASELINE_DAYS   = 7
CONTAMINATION   = 0.01
N_ESTIMATORS    = 300
MIN_SAMPLES_IF  = 150   

# --- Janelas de features ---
ROLL_5          = 5
ROLL_15         = 15
ROLL_60         = 60
ROLL_6H         = 360

# --- CUSUM ---
CUSUM_SLACK     = 0.5

# --- Health Index ---
HI_INIT             = 100.0
# Decaimento por anomalia de degradacao (IF)
HI_DECAY_IF_FAST    = 2.0    # score > 0.7
HI_DECAY_IF_MED     = 0.8    # score 0.5-0.7
HI_DECAY_IF_SLOW    = 0.2    # score < 0.5
# Decaimento por queda real (Ping == 0) — penalidade de outage
# Aplicado a cada minuto que o host está DOWN
HI_DECAY_DOWN       = 1.5    # pontos por minuto down
# Recuperacao por minuto saudavel (sem anomalia, Ping == 1)
HI_RECOVERY         = 0.05
HI_MIN              = 0.0
HI_MAX              = 100.0

RANDOM_STATE        = 42


# ===========
# 1. CARGA
# ===========
print("Carregando dados...")

df = pd.read_csv(CSV_PATH, low_memory=False)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "host_name"])

print(f"   Linhas brutas: {len(df):,} | Hosts unicos: {df['host_name'].nunique()}")


print("Preparando colunas de sinal...")

# Apenas a coluna "Ping" é usada como indicador de status (1=UP, 0=DOWN).
if "Ping" not in df.columns:
raise ValueError("Coluna 'Ping' não encontrada no CSV. Verifique o arquivo de entrada.")

df["ping_status"] = pd.to_numeric(df["Ping"], errors="coerce")

df["ping_status"] = df["ping_status"].apply(
lambda x: 1.0 if (pd.notna(x) and x > 0) else (0.0 if pd.notna(x) else np.nan)
)

df["Packet loss"]   = pd.to_numeric(df.get("Packet loss",   np.nan), errors="coerce")
df["Response time"] = pd.to_numeric(df.get("Response time", np.nan), errors="coerce")

df = df.sort_values(["host_name", "timestamp"])

n_up   = (df["ping_status"] == 1).sum()
n_down = (df["ping_status"] == 0).sum()
n_null = df["ping_status"].isna().sum()
print(f"   Ping -> UP: {n_up:,} | DOWN: {n_down:,} | Sem leitura: {n_null:,}")


# ==========================================================
# 3. REAMOSTRAGEM PARA 1 MINUTO POR HOST (TEMPO DE COLETA)
# ==========================================================
print("Reamostrando para 1 minuto...")

def resample_host(g):
g = g.set_index("timestamp")
out = g.resample(FREQ).agg({
    "ping_status":   "min", 
    "Packet loss":   "mean",
    "Response time": "mean",
})
out["host_name"] = g["host_name"].iloc[0]
return out.reset_index()

df_1m = (
df.groupby("host_name", group_keys=False)
.apply(resample_host)
.reset_index(drop=True)
)

# Preenche NaN de ping_status apos resample com 1.0
# (minutos sem coleta nao sao penalizados como queda)
df_1m["ping_status"] = df_1m["ping_status"].fillna(1.0)

print(f"   Total de linhas reamostradas: {len(df_1m):,}")


# =================================
# 4. DETECÇÃO DE QUEDAS (OUTAGES)
# =================================
print("Detectando quedas (ping_status == 0)...")

df_1m["is_down"] = (df_1m["ping_status"] == 0).astype(int)

outage_rows = []
for host, g in df_1m.groupby("host_name"):
g = g.sort_values("timestamp").copy()
g["down_shift"] = g["is_down"].diff().fillna(0)

starts = g.loc[g["down_shift"] == 1,  "timestamp"].tolist()
ends   = g.loc[g["down_shift"] == -1, "timestamp"].tolist()

if len(g) and g["is_down"].iloc[0] == 1:
starts = [g["timestamp"].iloc[0]] + starts
if len(starts) > len(ends):
ends = ends + [g["timestamp"].iloc[-1]]

for s, e in zip(starts, ends):
duration = (e - s).total_seconds() / 60.0
outage_rows.append({
    "host_name":    host,
    "start":        s,
    "end":          e,
    "duration_min": round(duration, 2),
})

outages = pd.DataFrame(outage_rows).sort_values("duration_min", ascending=False)
outages.to_csv(OUT_OUTAGES, index=False)
print(f"   Quedas detectadas: {len(outages):,} | Arquivo: {OUT_OUTAGES}")


# =================================================
# 5. ENGENHARIA DE FEATURES - APENAS PERÍODOS UP
# =================================================

FEATURE_COLS = [
"rt", "pl",
"rt_delta", "pl_delta", "rt_delta2",
"rt_roll_mean_5",  "rt_roll_mean_15", "rt_roll_mean_60", "rt_roll_mean_6h",
"pl_roll_mean_5",  "pl_roll_mean_15",
"rt_roll_std_15",  "rt_roll_std_60",
"pl_roll_std_15",
"pl_nonzero_rate_60", "pl_nonzero_rate_6h",
]

def build_features(g: pd.DataFrame) -> pd.DataFrame:
g = g.sort_values("timestamp").copy()
g_up = g[g["ping_status"] == 1].copy()

if len(g_up) < MIN_SAMPLES_IF:
return pd.DataFrame()

rt = g_up["Response time"].fillna(method="ffill").fillna(0)
pl = g_up["Packet loss"].fillna(0)

g_up["rt"] = rt
g_up["pl"] = pl
g_up["rt_delta"]  = rt.diff()
g_up["pl_delta"]  = pl.diff()
g_up["rt_delta2"] = g_up["rt_delta"].diff()

g_up["rt_roll_mean_5"]  = rt.rolling(ROLL_5,  min_periods=3).mean()
g_up["rt_roll_mean_15"] = rt.rolling(ROLL_15, min_periods=5).mean()
g_up["rt_roll_mean_60"] = rt.rolling(ROLL_60, min_periods=10).mean()
g_up["rt_roll_mean_6h"] = rt.rolling(ROLL_6H, min_periods=30).mean()
g_up["rt_roll_std_15"]  = rt.rolling(ROLL_15, min_periods=5).std()
g_up["rt_roll_std_60"]  = rt.rolling(ROLL_60, min_periods=10).std()

g_up["pl_roll_mean_5"]  = pl.rolling(ROLL_5,  min_periods=3).mean()
g_up["pl_roll_mean_15"] = pl.rolling(ROLL_15, min_periods=5).mean()
g_up["pl_roll_std_15"]  = pl.rolling(ROLL_15, min_periods=5).std()

g_up["pl_nonzero_rate_60"] = (pl > 0).rolling(ROLL_60, min_periods=10).mean()
g_up["pl_nonzero_rate_6h"] = (pl > 0).rolling(ROLL_6H, min_periods=30).mean()

return g_up


# ============
# 6. CUSUM
# ============

def compute_cusum(scores: np.ndarray, k: float = CUSUM_SLACK) -> np.ndarray:
"""
Acumula enquanto o score permanecer acima da referencia historica.
Nao cai enquanto a degradacao persistir.
"""
n = len(scores)
cusum = np.zeros(n)
mu_ref = np.median(scores)
for i in range(1, n):
cusum[i] = max(0.0, cusum[i-1] + (scores[i] - mu_ref) - k)
return cusum


# ==========================================================
# 7. HEALTH INDEX — com penalidade de outage (desconexão)
# ==========================================================

def compute_health_index_full(
timestamps:   pd.Series,
ping_series:  pd.Series,
if_timestamps: np.ndarray,
norm_scores:  np.ndarray,
is_anomaly:   np.ndarray,
cusum:        np.ndarray,
) -> np.ndarray:
"""
Calcula o Health Index minuto-a-minuto sobre TODOS os minutos do host
(UP e DOWN), integrando:
- Penalidade de outage : cada minuto com ping_status == 0 subtrai HI_DECAY_DOWN
- Penalidade IF        : minutos UP com anomalia detectada pelo IF
- Recuperacao          : minutos UP saudaveis, modulada pelo CUSUM
"""
n = len(timestamps)
hi = np.full(n, HI_INIT)

# Mapa timestamp -> score/anomalia/cusum para lookup rapido
score_map   = {}
anomaly_map = {}
cusum_map   = {}
for pos, ts in enumerate(if_timestamps):
score_map[ts]   = norm_scores[pos]
anomaly_map[ts] = is_anomaly[pos]
cusum_map[ts]   = cusum[pos]

cusum_max = cusum.max() if len(cusum) > 0 and cusum.max() > 0 else 1.0

ts_arr = timestamps.values

for i in range(1, n):
prev = hi[i - 1]
ts   = ts_arr[i]
ping = ping_series.iloc[i]

if ping == 0:
# Minuto de queda: penalidade fixa por outage
delta = -HI_DECAY_DOWN

elif ts in anomaly_map and anomaly_map[ts] == 1:
# Minuto UP com anomalia IF
s = score_map.get(ts, 0.0)
if s > 0.7:
delta = -HI_DECAY_IF_FAST
elif s > 0.5:
delta = -HI_DECAY_IF_MED
else:
delta = -HI_DECAY_IF_SLOW

else:
# Minuto saudavel: recuperacao modulada pelo CUSUM
cusum_val    = cusum_map.get(ts, 0.0)
cusum_norm_i = cusum_val / cusum_max
delta        = HI_RECOVERY * (1.0 - cusum_norm_i)

hi[i] = np.clip(prev + delta, HI_MIN, HI_MAX)

return hi


# =========================
# 8. PIPELINE PRINCIPAL
# =========================
print("\nExecutando pipeline por host...")
print(f"   Baseline: primeiros {BASELINE_DAYS} dias | Contamination IF: {CONTAMINATION}")
print("-" * 75)

all_results   = []
skipped_hosts = []

for host, g in df_1m.groupby("host_name"):
g = g.sort_values("timestamp").reset_index(drop=True)

total_min = len(g)
down_min  = (g["ping_status"] == 0).sum()
avail_pct = round(100.0 * (1 - down_min / total_min), 2) if total_min > 0 else 100.0

# --- Tenta construir features para o IF ---
g_feat = build_features(g)
has_if = not g_feat.empty

if has_if:
X_raw  = g_feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
has_if = len(X_raw) >= MIN_SAMPLES_IF

if has_if:
# Baseline
t_min         = g_feat.loc[X_raw.index, "timestamp"].min()
cutoff        = t_min + pd.Timedelta(days=BASELINE_DAYS)
baseline_mask = g_feat.loc[X_raw.index, "timestamp"] <= cutoff
X_baseline    = X_raw[baseline_mask]

if len(X_baseline) < MIN_SAMPLES_IF:
n_third    = max(MIN_SAMPLES_IF, len(X_raw) // 3)
X_baseline = X_raw.iloc[:n_third]

# Scaling (fit apenas no baseline)
scaler = RobustScaler()
scaler.fit(X_baseline)
Xs_all      = scaler.transform(X_raw)
Xs_baseline = scaler.transform(X_baseline)

# Isolation Forest
model = IsolationForest(
n_estimators=N_ESTIMATORS,
contamination=CONTAMINATION,
random_state=RANDOM_STATE,
n_jobs=-1,
)
model.fit(Xs_baseline)

raw_scores = -model.score_samples(Xs_all)
raw_preds  =  model.predict(Xs_all)

s_base        = -model.score_samples(Xs_baseline)
p_low, p_high = np.percentile(s_base, [5, 99])
norm_scores   = np.clip((raw_scores - p_low) / (p_high - p_low + 1e-9), 0, 1)
is_anomaly    = (raw_preds == -1).astype(int)
cusum_vals    = compute_cusum(norm_scores, k=CUSUM_SLACK)
if_ts         = g_feat.loc[X_raw.index, "timestamp"].values

else:
norm_scores = np.array([])
is_anomaly  = np.array([])
cusum_vals  = np.array([])
if_ts       = np.array([])
X_raw       = pd.DataFrame()
skipped_hosts.append(host)

# Health Index contabiliando em todo o período
hi_vals = compute_health_index_full(
timestamps    = g["timestamp"],
ping_series   = g["ping_status"],
if_timestamps = if_ts,
norm_scores   = norm_scores,
is_anomaly    = is_anomaly,
cusum         = cusum_vals,
)

# Resultado
result = g[["timestamp", "host_name", "ping_status",
"Packet loss", "Response time"]].copy()
result.rename(columns={"ping_status": "Ping"}, inplace=True)
result["is_down"]      = (result["Ping"] == 0).astype(int)
result["health_index"] = hi_vals

if has_if and len(X_raw) > 0:
if_df = pd.DataFrame({
    "timestamp":          g_feat.loc[X_raw.index, "timestamp"].values,
    "anomaly_score_norm": norm_scores,
    "is_anomaly":         is_anomaly,
    "cusum":              cusum_vals,
})
result = result.merge(if_df, on="timestamp", how="left")

valid         = result["anomaly_score_norm"].notna()
cusum_max_val = result.loc[valid, "cusum"].max()
cusum_max_val = cusum_max_val if (pd.notna(cusum_max_val) and cusum_max_val > 0) else 1.0
hi_drop       = (HI_INIT - result["health_index"]) / HI_INIT

result["criticality_index"] = np.nan
result.loc[valid, "criticality_index"] = np.clip(
0.35 * result.loc[valid, "anomaly_score_norm"]
+ 0.30 * (result.loc[valid, "cusum"] / cusum_max_val)
+ 0.35 * hi_drop[valid],
0, 1,
)
result.loc[~valid, "criticality_index"] = np.clip(hi_drop[~valid], 0, 1)
else:
result["anomaly_score_norm"] = np.nan
result["is_anomaly"]         = np.nan
result["cusum"]              = np.nan
hi_drop = (HI_INIT - result["health_index"]) / HI_INIT
result["criticality_index"]  = np.clip(hi_drop, 0, 1)

all_results.append(result)

hi_final  = hi_vals[-1]
crit_max  = result["criticality_index"].max()
n_anom    = int(is_anomaly.sum()) if len(is_anomaly) > 0 else 0
mode_flag = "IF+HI  " if has_if else "HI-only"
print(
f"   [{'OK' if has_if else '??'}] [{mode_flag}] {host:<42} | "
f"Disp: {avail_pct:>6.2f}% | Down: {down_min:>5}min | "
f"Anom: {n_anom:>4} | HI: {hi_final:>6.1f} | Crit: {crit_max:.3f}"
)

if skipped_hosts:
print(f"\n   {len(skipped_hosts)} hosts sem amostras UP suficientes para IF "
f"(HI calculado apenas por disponibilidade):")
for h in skipped_hosts:
print(f"      - {h}")


# ====================
# 9. EXPORTAÇÃO
# ====================
print("\nExportando resultados...")

if all_results:
full_df = pd.concat(all_results, ignore_index=True)

full_df.to_csv(OUT_HEALTH, index=False)
print(f"   Health Index completo : {OUT_HEALTH}")

anom_only = (
full_df[full_df["is_anomaly"] == 1]
.sort_values("criticality_index", ascending=False)
)
anom_only.to_csv(OUT_DEGRADATION, index=False)
print(f"   Anomalias (IF)        : {OUT_DEGRADATION}")

ranking = (
full_df.sort_values("timestamp")
.groupby("host_name")
.agg(
hi_final       = ("health_index",      "last"),
crit_max       = ("criticality_index", "max"),
cusum_max      = ("cusum",             "max"),
down_min_total = ("is_down",           "sum"),
total_min      = ("timestamp",         "count"),
)
.reset_index()
)
ranking["avail_pct"] = (
100.0 * (1 - ranking["down_min_total"] / ranking["total_min"])
).round(2)
ranking = ranking.sort_values("hi_final")

print("\nRANKING DE CRITICIDADE (menor HI = mais critico):")
print(ranking[["host_name", "hi_final", "crit_max",
"down_min_total", "avail_pct"]].to_string(index=False))

ranking.to_csv(OUT_RANKING, index=False)
print(f"\n   Ranking exportado     : {OUT_RANKING}")

outages.to_csv(OUT_OUTAGES, index=False)
print(f"   Outages exportados    : {OUT_OUTAGES}")
print("\nPipeline concluido com sucesso!")