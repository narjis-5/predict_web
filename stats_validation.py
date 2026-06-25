# -*- coding: utf-8 -*-
"""Validation statistique du modele de regression (duree) et du classifieur
(classe de duree). Tout provient des deux bundles .joblib et du parquet.

- CV stratifiee 5 plis (classification) / KFold 5 plis (regression).
- Bootstrap 2000 reechantillons sur le TEST scelle -> IC 95 %.
- Test de DeLong : AUC LogReg vs RandomForest.
- Test de McNemar : LogReg vs RandomForest (les deux meilleurs par exactitude/F1).

N'invente rien : LogReg et le modele de regression sont charges depuis les
bundles ; RandomForest est RE-ENTRAINE avec les hyperparametres exacts documentes
dans ml_academique/tables/clf_comparatif.csv (section 17 du notebook), car aucun
bundle RF n'a ete serialise.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             accuracy_score, f1_score, make_scorer)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as st

HERE = Path(r"C:\Users\user\Desktop\pfeN")
RS = 42
NUM = ["quantity", "shipping_cost_ngn", "estimated_transit_days"]
OH = ["supplier_name", "logistics_company", "ship_month", "ship_day_of_week"]
TE = ["origin_city", "destination_city"]

# --------------------------------------------------------------------------
# 0. Charger bundles + parquet, reconstruire les variables (anti-fuite)
# --------------------------------------------------------------------------
reg_bundle = joblib.load(HERE / "delivery_duration_model_bundle.joblib")
clf_bundle = joblib.load(HERE / "delivery_duration_class_model_bundle.joblib")
reg_pipe = reg_bundle["pipeline"]
clf_pipe = clf_bundle["pipeline"]

df = pd.read_parquet(HERE / "nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet")
for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df = df.dropna(subset=["ship_date", "expected_delivery_date", "actual_delivery_date"]).copy()
df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days
df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)
df = df[(df["estimated_transit_days"] >= 0) & (df["realized_transit_days"] >= 0)].copy()
df = df.dropna(subset=["estimated_transit_days", "realized_transit_days"]).copy()

X = df[NUM + OH + TE].copy()

# Partition REGRESSION (cible continue, non stratifiee) : identique au notebook
y_reg = df["realized_transit_days"].astype(float)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.20, random_state=RS)

# Partition CLASSIFICATION (classe de duree, stratifiee) : identique a make_class_bundle
med = float(df["realized_transit_days"].median())
y_clf = (df["realized_transit_days"] > med).astype(int)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_clf, test_size=0.20, random_state=RS, stratify=y_clf)

print(f"Seuil mediane = {med:.1f} j | Test regression n={len(Xr_te)} | Test classif n={len(Xc_te)}")

rows = []  # lignes du tableau recapitulatif

# --------------------------------------------------------------------------
# 1. Validation croisee 5 plis
# --------------------------------------------------------------------------
print("\n[1] Validation croisee 5 plis ...")
# Regression : KFold (stratification non applicable a une cible continue)
kf = KFold(n_splits=5, shuffle=True, random_state=RS)
r2_cv = cross_val_score(clone(reg_pipe), Xr_tr, yr_tr, scoring="r2", cv=kf, n_jobs=-1)
rows.append(dict(analyse="CV_5plis", modele="HistGBR_regression", metrique="R2",
                 valeur=r2_cv.mean(), ecart_type=r2_cv.std(), ic95_bas="", ic95_haut="", p_value=""))
print(f"  Regression  R2  = {r2_cv.mean():.4f} ± {r2_cv.std():.4f}")

# Regression (variante) : KFold stratifie sur les DECILES de la duree, pour
# garantir une distribution de duree homogene entre plis (cible continue binnee).
bins = pd.qcut(yr_tr, q=10, labels=False, duplicates="drop")
skf_reg = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
r2_strat = []
for tr_idx, va_idx in skf_reg.split(Xr_tr, bins):
    m = clone(reg_pipe).fit(Xr_tr.iloc[tr_idx], yr_tr.iloc[tr_idx])
    r2_strat.append(r2_score(yr_tr.iloc[va_idx], m.predict(Xr_tr.iloc[va_idx])))
r2_strat = np.array(r2_strat)
rows.append(dict(analyse="CV_5plis_strat_deciles", modele="HistGBR_regression", metrique="R2",
                 valeur=r2_strat.mean(), ecart_type=r2_strat.std(), ic95_bas="", ic95_haut="", p_value=""))
print(f"  Regression  R2 (strat. deciles) = {r2_strat.mean():.4f} ± {r2_strat.std():.4f}")

# Classification : StratifiedKFold, AUC / exactitude / F1-macro
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
scorers = {"AUC": "roc_auc", "exactitude": "accuracy", "F1_macro": "f1_macro"}
for nom, sc in scorers.items():
    s = cross_val_score(clone(clf_pipe), Xc_tr, yc_tr, scoring=sc, cv=skf, n_jobs=-1)
    rows.append(dict(analyse="CV_5plis", modele="LogReg_classification", metrique=nom,
                     valeur=s.mean(), ecart_type=s.std(), ic95_bas="", ic95_haut="", p_value=""))
    print(f"  Classif {nom:11s} = {s.mean():.4f} ± {s.std():.4f}")

# --------------------------------------------------------------------------
# 2. Bootstrap 2000 reechantillons sur le TEST -> IC 95 %
# --------------------------------------------------------------------------
print("\n[2] Bootstrap 2000 reechantillons (test scelle) ...")
B = 2000
rng = np.random.default_rng(RS)

# Predictions point sur le test (calculees une seule fois)
pred_reg = reg_pipe.predict(Xr_te)
yr_te_a = yr_te.to_numpy()
proba_lr = clf_pipe.predict_proba(Xc_te)[:, 1]
pred_lr = clf_pipe.predict(Xc_te)
yc_te_a = yc_te.to_numpy()

def boot_ci(metric_fn, n, *arrays):
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = metric_fn(*[a[idx] for a in arrays])
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

nr = len(yr_te_a)
r2_pt = r2_score(yr_te_a, pred_reg)
r2_lo, r2_hi = boot_ci(lambda yt, yp: r2_score(yt, yp), nr, yr_te_a, pred_reg)
rows.append(dict(analyse="Bootstrap_IC95", modele="HistGBR_regression", metrique="R2",
                 valeur=r2_pt, ecart_type="", ic95_bas=r2_lo, ic95_haut=r2_hi, p_value=""))
mae_pt = mean_absolute_error(yr_te_a, pred_reg)
mae_lo, mae_hi = boot_ci(lambda yt, yp: mean_absolute_error(yt, yp), nr, yr_te_a, pred_reg)
rows.append(dict(analyse="Bootstrap_IC95", modele="HistGBR_regression", metrique="MAE_j",
                 valeur=mae_pt, ecart_type="", ic95_bas=mae_lo, ic95_haut=mae_hi, p_value=""))
print(f"  R2  = {r2_pt:.4f}  IC95 [{r2_lo:.4f}, {r2_hi:.4f}]")
print(f"  MAE = {mae_pt:.4f}  IC95 [{mae_lo:.4f}, {mae_hi:.4f}]")

nc = len(yc_te_a)
auc_pt = roc_auc_score(yc_te_a, proba_lr)
auc_lo, auc_hi = boot_ci(lambda yt, yp: roc_auc_score(yt, yp), nc, yc_te_a, proba_lr)
rows.append(dict(analyse="Bootstrap_IC95", modele="LogReg_classification", metrique="AUC",
                 valeur=auc_pt, ecart_type="", ic95_bas=auc_lo, ic95_haut=auc_hi, p_value=""))
acc_pt = accuracy_score(yc_te_a, pred_lr)
acc_lo, acc_hi = boot_ci(lambda yt, yp: accuracy_score(yt, yp), nc, yc_te_a, pred_lr)
rows.append(dict(analyse="Bootstrap_IC95", modele="LogReg_classification", metrique="exactitude",
                 valeur=acc_pt, ecart_type="", ic95_bas=acc_lo, ic95_haut=acc_hi, p_value=""))
print(f"  AUC = {auc_pt:.4f}  IC95 [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"  Acc = {acc_pt:.4f}  IC95 [{acc_lo:.4f}, {acc_hi:.4f}]")

# --------------------------------------------------------------------------
# 3. Re-entrainer RandomForest (params documentes) pour les comparaisons
# --------------------------------------------------------------------------
print("\n[3] Re-entrainement RandomForest (params clf_comparatif.csv) ...")
pre = clone(clf_pipe.named_steps["pre"])  # meme preprocessing anti-fuite
rf_pipe = Pipeline([("pre", pre), ("m", RandomForestClassifier(
    n_estimators=250, min_samples_leaf=20, max_features=0.5, max_depth=20,
    n_jobs=-1, random_state=RS))]).fit(Xc_tr, yc_tr)
proba_rf = rf_pipe.predict_proba(Xc_te)[:, 1]
pred_rf = rf_pipe.predict(Xc_te)
auc_lr = roc_auc_score(yc_te_a, proba_lr)
auc_rf = roc_auc_score(yc_te_a, proba_rf)
print(f"  AUC LogReg = {auc_lr:.4f} | AUC RF = {auc_rf:.4f}")

# --------------------------------------------------------------------------
# 4. Test de DeLong (AUC LogReg vs RandomForest) -- implementation rapide
# --------------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_test(y_true, p1, p2):
    y_true = np.asarray(y_true); order = np.argsort(-y_true)
    m = int(y_true.sum())
    preds = np.vstack((p1, p2))[:, order]
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * st.norm.sf(abs(z))
    return aucs, float(z), float(p)

print("\n[4] Test de DeLong ...")
(auc_d_lr, auc_d_rf), z_delong, p_delong = delong_test(yc_te_a, proba_lr, proba_rf)
rows.append(dict(analyse="DeLong_AUC", modele="LogReg_vs_RandomForest",
                 metrique=f"AUC_LR={auc_d_lr:.4f}; AUC_RF={auc_d_rf:.4f}; z={z_delong:.3f}",
                 valeur=auc_d_lr - auc_d_rf, ecart_type="", ic95_bas="", ic95_haut="", p_value=p_delong))
print(f"  delta AUC = {auc_d_lr - auc_d_rf:+.5f} | z = {z_delong:.3f} | p = {p_delong:.4f}")

# --------------------------------------------------------------------------
# 5. Test de McNemar (LogReg vs RandomForest sur le test)
# --------------------------------------------------------------------------
print("\n[5] Test de McNemar ...")
c1 = (pred_lr == yc_te_a)   # LogReg correct
c2 = (pred_rf == yc_te_a)   # RF correct
b = int(np.sum(c1 & ~c2))   # LR bon, RF faux
c = int(np.sum(~c1 & c2))   # LR faux, RF bon
nd = b + c
# p-valeur exacte (binomiale, bilaterale) + statistique chi2 corrigee
p_exact = float(min(1.0, 2 * st.binom.cdf(min(b, c), nd, 0.5))) if nd > 0 else 1.0
chi2_cc = float((abs(b - c) - 1) ** 2 / nd) if nd > 0 else 0.0
rows.append(dict(analyse="McNemar", modele="LogReg_vs_RandomForest",
                 metrique=f"b(LR+,RF-)={b}; c(LR-,RF+)={c}; chi2_cc={chi2_cc:.3f}",
                 valeur=b - c, ecart_type="", ic95_bas="", ic95_haut="", p_value=p_exact))
print(f"  b={b} | c={c} | chi2(cc)={chi2_cc:.3f} | p_exact={p_exact:.4f}")

# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
out = pd.DataFrame(rows, columns=["analyse", "modele", "metrique", "valeur",
                                  "ecart_type", "ic95_bas", "ic95_haut", "p_value"])
for col in ["valeur", "ecart_type", "ic95_bas", "ic95_haut", "p_value"]:
    out[col] = pd.to_numeric(out[col].replace("", np.nan))
dest = HERE / "ml_academique" / "tables" / "stats_validation.csv"
out.to_csv(dest, index=False, encoding="utf-8")
pd.set_option("display.width", 200, "display.max_columns", None)
print("\n===== Tableau recapitulatif =====")
print(out.to_string(index=False))
print(f"\nExporte -> {dest}")
