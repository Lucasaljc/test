import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# =====================================================================
# 0) EXEMPLE DE DATASET (à remplacer par ton vrai df)
# =====================================================================
dates = pd.date_range("2023-01-01", periods=300, freq="D")
rng = np.random.default_rng(0)

df = pd.DataFrame({
    "date": dates,
    "colonne_1": np.sin(np.arange(300) / 20) + rng.normal(scale=0.3, size=300),  # target
    "colonne_2": rng.normal(size=300),                                           # feature
    "colonne_3": rng.normal(size=300),                                           # feature
})

# On crée quelques NaN pour tester le pipeline
df.loc[::10, "colonne_1"] = np.nan
df.loc[::15, "colonne_2"] = np.nan
df.loc[::20, "colonne_3"] = np.nan

# =====================================================================
# 1) Préparation date + index temporel + NaN plots
# =====================================================================

date_col   = "date"
target_col = "colonne_1"

# Date -> datetime
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df = df.sort_values(date_col).set_index(date_col)

# --- 1.1 Proportion de NaN par colonne ---
na_ratio = df.isna().mean()

plt.figure(figsize=(8, 4))
na_ratio.plot(kind="bar")
plt.ylabel("Proportion de NaN")
plt.title("Proportion de valeurs manquantes par colonne")
plt.tight_layout()
plt.show()

# --- 1.2 Localisation des NaN dans le temps ---
plt.figure(figsize=(10, 4))
plt.imshow(df.isna().T, aspect="auto", interpolation="nearest")
plt.yticks(range(len(df.columns)), df.columns)
plt.xlabel("Temps (index)")
plt.title("Carte des NaN (True = NaN)")
plt.colorbar(label="NaN")
plt.tight_layout()
plt.show()

# =====================================================================
# 2) Complétion des dates et forward fill
# =====================================================================

# On suppose une fréquence journalière ; à adapter si besoin
df = df.asfreq("D")
df = df.ffill()  # forward fill (on peut ajouter .bfill() si nécessaire)

daily = df.copy()

# =====================================================================
# 3) Création des features : Y, lags (7, 14, 21, 28)
# =====================================================================

# Target
daily["Y"] = daily[target_col]

# Lags de la target
for lag in [7, 14, 21, 28]:
    daily[f"Y_lag{lag}"] = daily["Y"].shift(lag)

# Lags des autres features (toutes les colonnes sauf target+Y)
feature_cols_raw = [c for c in daily.columns if c not in [target_col, "Y"]]
for lag in [7, 14, 21, 28]:
    for c in feature_cols_raw:
        daily[f"{c}_lag{lag}"] = daily[c].shift(lag)

# =====================================================================
# 4) Walk-forward prediction hebdomadaire
# =====================================================================

def walk_forward_pred(df_day, feat_cols, model, weeks_train=20, purge_days=7):
    """
    Walk-forward hebdomadaire : on entraîne sur 'weeks_train' semaines,
    on purge 'purge_days' jours avant la fenêtre de prévision,
    et on prédit du lundi au dimanche.
    """
    df_loc = df_day.sort_index()
    if not isinstance(df_loc.index, pd.DatetimeIndex):
        df_loc.index = pd.to_datetime(df_loc.index, errors="coerce")
        df_loc = df_loc.dropna(subset=[df_loc.index.name])
        df_loc = df_loc.sort_index()

    mondays = pd.date_range(df_loc.index.min(), df_loc.index.max(), freq="W-MON")

    yhat = pd.Series(index=df_loc.index, dtype=float)

    for i in range(weeks_train + 1, len(mondays)):
        start_pred = mondays[i]
        end_pred   = start_pred + pd.Timedelta(days=6)
        if end_pred > df_loc.index.max():
            break

        # Fenêtre d'entraînement
        train_end   = start_pred - pd.Timedelta(days=purge_days)
        train_start = train_end - pd.Timedelta(days=7*weeks_train - 1)

        cols  = ["Y"] + feat_cols
        train = df_loc.loc[train_start:train_end, cols].dropna()
        if len(train) < 50:
            continue

        fitted = clone(model).fit(train[feat_cols].values, train["Y"].values)

        # Fenêtre de test (semaine suivante)
        test = df_loc.loc[start_pred:end_pred, cols].dropna()
        if test.empty:
            continue

        yhat.loc[test.index] = fitted.predict(test[feat_cols].values)

    if len(mondays) > weeks_train + 1:
        eval_start = mondays[weeks_train + 1]
    else:
        eval_start = df_loc.index.min()

    return yhat, eval_start

# =====================================================================
# 5) Métriques (robuste si pas d'échantillons)
# =====================================================================

def metrics(y_true, y_pred):
    idx = y_true.index.intersection(y_pred.index)
    if len(idx) == 0:
        # Aucun recouvrement
        return np.nan, np.nan, y_true.iloc[0:0], y_pred.iloc[0:0]

    y_t = y_true.loc[idx].astype(float)
    y_p = y_pred.loc[idx].astype(float)
    mask = (~y_t.isna()) & (~y_p.isna())
    y_t, y_p = y_t[mask], y_p[mask]

    if len(y_t) == 0:
        return np.nan, np.nan, y_t, y_p

    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2   = r2_score(y_t, y_p)
    return rmse, r2, y_t, y_p

# =====================================================================
# 6) Modèles + prédictions (comparaison baseline / linéaire / Ridge)
# =====================================================================

model_lin  = LinearRegression()
model_r10  = Ridge(alpha=10)
model_r1e4 = Ridge(alpha=10000)

# Baseline naïve : Y(t-7)
baseline = daily["Y"].shift(7)

# 1) Linear Regression (Y_lag7)
feat_m0 = ["Y_lag7"]
pred_m0, eval_start = walk_forward_pred(daily, feat_cols=feat_m0, model=model_lin)

# 2) Linear Regression (Y_lag7 + Y_lag14 + Y_lag21 + Y_lag28)
feat_mlags = ["Y_lag7", "Y_lag14", "Y_lag21", "Y_lag28"]
pred_md, _ = walk_forward_pred(daily, feat_cols=feat_mlags, model=model_lin)

# 3) Ridge Regression (Y_lag7 + Y_lag14 + Y_lag21 + Y_lag28)
pred_md2, _ = walk_forward_pred(daily, feat_cols=feat_mlags, model=model_r10)

# 4) Ridge Regression (Y_lag7 + lags des features)
lag_feat_cols = [
    c for c in daily.columns
    if any(c.endswith(f"_lag{lag}") for lag in [7, 14, 21, 28])
    and not c.startswith("Y_lag")
]
feat_me = ["Y_lag7", "Y_lag14", "Y_lag21", "Y_lag28"] + lag_feat_cols
pred_me, _ = walk_forward_pred(daily, feat_cols=feat_me, model=model_r1e4)

# =====================================================================
# 7) Évaluation + plots comparatifs (comme ton code initial)
# =====================================================================

y_true_eval = daily["Y"][daily.index >= eval_start]
baseline    = baseline[baseline.index >= eval_start]
pred_m0     = pred_m0[pred_m0.index >= eval_start]
pred_md     = pred_md[pred_md.index >= eval_start]
pred_md2    = pred_md2[pred_md2.index >= eval_start]
pred_me     = pred_me[pred_me.index >= eval_start]

models = [
    ("Baseline (lag-7)",                        baseline),
    ("Linear Regression (Y_lag7)",              pred_m0),
    ("Linear Regression (Y_lag7,14,21,28)",     pred_md),
    ("Ridge (Y_lag7,14,21,28)",                 pred_md2),
    ("Ridge (Y_lag7,14,21,28 + lags feats)",    pred_me),
]

metrics_list = []
series_list  = []

print("=== Comparaison modèles ===")
for name, pred in models:
    rmse, r2, yt, yp = metrics(y_true_eval, pred)
    metrics_list.append((name, rmse, r2))
    series_list.append((yt, yp))
    print(f"{name:40s} -> RMSE: {rmse:,.3f} | R²: {r2: .3f}")

# Plots temporels
start_plot, end_plot = eval_start, daily.index.max()
fig, axes = plt.subplots(len(models), 1, figsize=(16, 10), sharex=True)

for ax, (name, rmse, r2), (yt, yp) in zip(axes, metrics_list, series_list):
    yt = yt.loc[start_plot:end_plot]
    yp = yp.loc[start_plot:end_plot]
    ax.plot(yt.index, yt.values, label="True")
    ax.plot(yp.index, yp.values, label="Pred")
    ax.set_title(f"{name} — RMSE={rmse:,.3f} | R²={r2 if not np.isnan(r2) else np.nan:.2f}")
    ax.set_ylabel("Target")
    ax.legend()

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.show()

# =====================================================================
# 8) RMSE en fonction de lambda (Ridge) + coefficients vs lambda
# =====================================================================

# On utilise le modèle le plus riche pour l'analyse de lambda
# (feat_me). On fait :
#   - walk-forward pour la RMSE en fonction de alpha
#   - fit "global" sur toutes les données pour tracer les coefficients

# Matrice complète pour les coefficients
X_all = daily[feat_me].dropna()
y_all = daily.loc[X_all.index, "Y"]

alphas = np.logspace(0, 5, 30)  # de 10^0 à 10^5 (log-scale, pas de 0)
rmse_alphas = []
coefs = []

for a in alphas:
    # --- Coeffs globaux ---
    model_global = Ridge(alpha=a)
    model_global.fit(X_all.values, y_all.values)
    coefs.append(model_global.coef_)

    # --- RMSE walk-forward ---
    model_ridge_a = Ridge(alpha=a)
    pred_a, eval_start_a = walk_forward_pred(daily, feat_cols=feat_me,
                                             model=model_ridge_a,
                                             weeks_train=20, purge_days=7)
    y_true_eval_a = daily["Y"][daily.index >= eval_start_a]
    pred_a = pred_a[pred_a.index >= eval_start_a]
    rmse_a, _, _, _ = metrics(y_true_eval_a, pred_a)
    rmse_alphas.append(rmse_a)

rmse_alphas = np.array(rmse_alphas, dtype=float)
coefs = np.array(coefs)  # shape = (n_alphas, n_features)

# On ignore les NaN pour trouver le meilleur lambda
valid_mask = ~np.isnan(rmse_alphas)
best_idx = np.argmin(rmse_alphas[valid_mask])
best_alpha = alphas[valid_mask][best_idx]
best_rmse = rmse_alphas[valid_mask][best_idx]

print("\n=== Analyse Ridge vs lambda ===")
print(f"Meilleur alpha (RMSE minimale) : {best_alpha:.4g}  |  RMSE = {best_rmse:,.4f}")

# --- Plot RMSE vs lambda ---
plt.figure(figsize=(8, 4))
plt.semilogx(alphas, rmse_alphas, marker="o")
plt.axvline(best_alpha, linestyle="--", label=f"alpha* = {best_alpha:.3g}")
plt.xlabel("alpha (lambda, échelle log)")
plt.ylabel("RMSE (walk-forward)")
plt.title("RMSE en fonction de lambda (Ridge)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot coefficients vs lambda ---
plt.figure(figsize=(10, 6))
for j, feat_name in enumerate(feat_me):
    plt.semilogx(alphas, coefs[:, j], label=feat_name)
plt.axvline(best_alpha, linestyle="--", label=f"alpha* = {best_alpha:.3g}")
plt.xlabel("alpha (lambda, échelle log)")
plt.ylabel("Coefficients")
plt.title("Coefficients Ridge en fonction de lambda")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =====================================================================
# 9) RMSE moyenne en fonction du nombre d'observations de train
# =====================================================================

weeks_list = [4, 6, 8, 10, 12, 16, 20, 24]
rmse_weeks = []
nobs_weeks = []

for w in weeks_list:
    model_ridge_w = Ridge(alpha=best_alpha)
    pred_w, eval_start_w = walk_forward_pred(
        daily, feat_cols=feat_me, model=model_ridge_w,
        weeks_train=w, purge_days=7
    )
    y_true_eval_w = daily["Y"][daily.index >= eval_start_w]
    pred_w = pred_w[pred_w.index >= eval_start_w]
    rmse_w, _, _, _ = metrics(y_true_eval_w, pred_w)
    rmse_weeks.append(rmse_w)
    nobs_weeks.append(w * 7)  # nb d'observations de train (semaines * 7 jours)

rmse_weeks = np.array(rmse_weeks, dtype=float)
nobs_weeks = np.array(nobs_weeks, dtype=int)

valid_mask_w = ~np.isnan(rmse_weeks)

plt.figure(figsize=(8, 4))
plt.plot(nobs_weeks[valid_mask_w], rmse_weeks[valid_mask_w], marker="o")
plt.xlabel("Nombre d'observations utilisées pour le train (~ semaines * 7)")
plt.ylabel("RMSE (walk-forward)")
plt.title("RMSE moyenne en fonction de la taille de la fenêtre de train")
plt.tight_layout()
plt.show()
# test
