
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix

TARGET_HOST = "XXX_SUB_IBY_VST_01"

# -----------------------------
# FEATURES
# -----------------------------
feature_cols = [
    "rt", "pl",
    "rt_delta", "pl_delta",
    "rt_roll_mean_15", "rt_roll_std_15",
    "pl_roll_mean_15",
    "pl_nonzero_rate_60"
]

# -----------------------------
# Filtra host e monta dataset
# -----------------------------
g = df_1m[df_1m["host_name"] == TARGET_HOST].sort_values("timestamp").copy()

# mantém só períodos UP (Ping == 1) como você já faz
g_up = g[g["Ping"] == 1].copy()
if len(g_up) < 200:
    raise ValueError(f"Host {TARGET_HOST}: poucas amostras UP ({len(g_up)}).")

# base signals
g_up["rt"] = g_up["Response time"]
g_up["pl"] = g_up["Packet loss"]

# deltas
g_up["rt_delta"] = g_up["rt"].diff()
g_up["pl_delta"] = g_up["pl"].diff()

# rolling features
g_up["rt_roll_mean_15"] = g_up["rt"].rolling(ROLL_15, min_periods=5).mean()
g_up["rt_roll_std_15"]  = g_up["rt"].rolling(ROLL_15, min_periods=5).std()
g_up["pl_roll_mean_15"] = g_up["pl"].rolling(ROLL_15, min_periods=5).mean()
g_up["pl_nonzero_rate_60"] = (g_up["pl"] > 0).rolling(ROLL_60, min_periods=10).mean()

# X original (unidades reais)
X_raw = g_up[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
if len(X_raw) < 200:
    raise ValueError(f"Host {TARGET_HOST}: após dropna, sobraram {len(X_raw)} linhas.")

# scaling robusto 
X_med = X_raw.median()
X_iqr = (X_raw.quantile(0.75) - X_raw.quantile(0.25)).replace(0, 1.0)
Xs = (X_raw - X_med) / X_iqr

# -----------------------------
# 1) TREINAMENTO ISOLATION
# -----------------------------
if_model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
if_model.fit(Xs)

preds = if_model.predict(Xs)                  # -1 anomalia, +1 normal
scores = -if_model.score_samples(Xs)          # maior = mais anômalo
y = (preds == -1).astype(int)                 # 1 anomalia, 0 normal

print("Taxa de anomalias (IF):", y.mean())

# -----------------------------
# Treina árvore substituta (surrogate) 
# -----------------------------
surrogate = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=25,
    class_weight="balanced",    # ajuda porque anomalia é rara
    random_state=RANDOM_STATE
)
surrogate.fit(X_raw, y)

# Fidelidade: quão bem a árvore imita o IF
fidelity = surrogate.score(X_raw, y)
print("Fidelidade da surrogate vs IsolationForest:", fidelity)

# -----------------------------
# 3) SURROGATE TREE
# -----------------------------
plt.figure(figsize=(26, 12))
plot_tree(
    surrogate,
    feature_names=feature_cols,
    class_names=["Normal", "Anomalia"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title(f"Surrogate Decision Tree - Estudo de caso")
plt.tight_layout()
plt.show()


rules = export_text(surrogate, feature_names=feature_cols)
print("\nREGRAS DA ÁRVORE (texto):\n")
print(rules)

y_hat = surrogate.predict(X_raw)

print("\nMatriz de confusão (surrogate vs IF):\n", confusion_matrix(y, y_hat))
print("\nRelatório (surrogate vs IF):\n", classification_report(y, y_hat, target_names=["Normal", "Anomalia"]))
