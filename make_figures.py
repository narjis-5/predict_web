# -*- coding: utf-8 -*-
"""Genere TOUTES les figures de resultats du memoire dans ml_academique/figures/.
Tout provient des modeles (2 bundles) et des donnees du TEST SCELLE. Aucune metrique inventee.
"""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             r2_score, mean_absolute_error)

RS = 42; DPI = 160
sns.set_theme(style="whitegrid", context="notebook")
HERE = Path("."); FIG = HERE / "ml_academique" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, RED = "#1f6f9f", "#e8743b", "#1f9d6b", "#d24b54"

# ---------------------------------------------------------------- data + bundles
reg = joblib.load("delivery_duration_model_bundle.joblib")
clf = joblib.load("delivery_duration_class_model_bundle.joblib")
SEUIL = float(clf["seuil_jours"])
NUM = ["quantity", "shipping_cost_ngn", "estimated_transit_days"]
OH = ["supplier_name", "logistics_company", "ship_month", "ship_day_of_week"]
TE = ["origin_city", "destination_city"]

df = pd.read_parquet("nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet")
for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df = df.dropna(subset=["ship_date", "expected_delivery_date", "actual_delivery_date"]).copy()
df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days
df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)
df = df[(df["estimated_transit_days"] >= 0) & (df["realized_transit_days"] >= 0)].copy()
df["is_overrun"] = (df["realized_transit_days"] > df["estimated_transit_days"]).astype(int)

X = df[NUM + OH + TE].copy()
y_reg = df["realized_transit_days"].astype(float)
y_clf = (df["realized_transit_days"] > SEUIL).astype(int)

# Split REGRESSION (comme le notebook : random_state=42, sans stratification)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.20, random_state=RS)
# Split CLASSIFICATION (comme le bundle : stratifie)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_clf, test_size=0.20, random_state=RS, stratify=y_clf)

def make_pre(ttype):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("oh", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("e", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), OH),
        ("te", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("e", TargetEncoder(target_type=ttype, cv=5, random_state=RS))]), TE),
    ])

created = []
def save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    created.append(name); print("OK", name)

# ====================================================== 1. eda.png (Figure A.1)
try:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].hist(df["estimated_transit_days"], bins=30, alpha=.65, label="planifiee (ETA)", color=BLUE)
    ax[0].hist(df["realized_transit_days"], bins=30, alpha=.65, label="reelle", color=ORANGE)
    ax[0].legend(); ax[0].set_title("Distribution des durees : reelle vs planifiee")
    ax[0].set_xlabel("jours"); ax[0].set_ylabel("nombre d'expeditions")
    s = df.sample(min(20000, len(df)), random_state=RS)
    ax[1].scatter(s["estimated_transit_days"], s["realized_transit_days"], s=5, alpha=.12, color=BLUE)
    lim = [0, df["realized_transit_days"].quantile(.99)]
    ax[1].plot(lim, lim, "--", color=RED); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_title("Duree planifiee vs reelle"); ax[1].set_xlabel("planifiee (j)"); ax[1].set_ylabel("reelle (j)")
    rate = df.groupby("logistics_company")["is_overrun"].mean().sort_values()
    ax[2].barh(rate.index, rate.values, color="#7c93b8")
    ax[2].set_title("Taux de depassement de l'ETA par transporteur"); ax[2].set_xlabel("taux de depassement")
    fig.suptitle("Figure A.1 — Analyse exploratoire des durees de livraison", fontsize=13, fontweight="bold")
    save(fig, "eda.png")
except Exception as e:
    print("FAIL eda.png:", e)

# =========================== 2&3. classifieurs : LogReg(bundle) + RF + ExtraTrees
clf_models = {}
try:
    # LogReg : on UTILISE le bundle (pas de reentrainement)
    clf_models["LogReg"] = clf["pipeline"]
    # RandomForest et ExtraTrees : memes preprocessing + split que le notebook (sect. 17.3)
    rf = Pipeline([("pre", make_pre("binary")),
                   ("m", RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_leaf=20,
                                                class_weight="balanced", n_jobs=-1, random_state=RS))]).fit(Xc_tr, yc_tr)
    et = Pipeline([("pre", make_pre("binary")),
                   ("m", ExtraTreesClassifier(n_estimators=250, max_depth=20, min_samples_leaf=20,
                                              class_weight="balanced", n_jobs=-1, random_state=RS))]).fit(Xc_tr, yc_tr)
    clf_models["RandomForest"] = rf
    clf_models["ExtraTrees"] = et

    labels = ["standard (<=%d j)" % SEUIL, "longue (>%d j)" % SEUIL]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (nm, mdl) in zip(axes, clf_models.items()):
        pr = mdl.predict(Xc_te)
        cm = confusion_matrix(yc_te, pr, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 13})
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        ax.set_title(f"{nm}  (exactitude = {acc:.3f})"); ax.set_xlabel("predit"); ax.set_ylabel("reel")
        if nm == "LogReg":
            print("   LogReg confusion tn/fp/fn/tp =", cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    fig.suptitle("Figure 3.1 — Matrices de confusion des 3 meilleurs classifieurs (test scelle)",
                 fontsize=13, fontweight="bold")
    save(fig, "clf_confusions.png")

    # ROC superposees
    fig, ax = plt.subplots(figsize=(7, 6.2))
    for nm, mdl in clf_models.items():
        proba = mdl.predict_proba(Xc_te)[:, 1]
        fpr, tpr, _ = roc_curve(yc_te, proba); auc = roc_auc_score(yc_te, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{nm} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#888")
    ax.set_xlabel("Taux de faux positifs"); ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Figure 3.2 — Courbes ROC des 3 meilleurs classifieurs", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    save(fig, "clf_roc.png")
except Exception as e:
    print("FAIL clf figures:", e)

# ====================================== 4. feature_selection.png (RF reg, top12)
try:
    pre_r = reg["pipeline"].named_steps["pre"]
    feat = pre_r.get_feature_names_out()
    idx = Xr_tr.sample(min(50000, len(Xr_tr)), random_state=RS).index
    Xt = pre_r.transform(Xr_tr.loc[idx]); yt = yr_tr.loc[idx].to_numpy()
    rfr = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=20,
                                n_jobs=-1, random_state=RS).fit(Xt, yt)
    imp = pd.Series(rfr.feature_importances_, index=feat).sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_title("Figure A.2 — Importance des variables (foret aleatoire, top 12)", fontsize=12, fontweight="bold")
    ax.set_xlabel("importance par reduction d'impurete")
    save(fig, "feature_selection.png")
except Exception as e:
    print("FAIL feature_selection.png:", e)

# ====================================== 5. diagnostics.png (predit vs reel + residus)
try:
    pred_r = reg["pipeline"].predict(Xr_te)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].scatter(yr_te, pred_r, s=5, alpha=.15, color=BLUE)
    lim = [min(yr_te.min(), pred_r.min()), max(yr_te.max(), pred_r.max())]
    ax[0].plot(lim, lim, "--", color=RED)
    ax[0].set_xlabel("duree reelle (j)"); ax[0].set_ylabel("duree predite (j)")
    mae = mean_absolute_error(yr_te, pred_r); r2 = r2_score(yr_te, pred_r)
    ax[0].set_title(f"Predit vs reel  (R2 = {r2:.3f}, MAE = {mae:.2f} j)")
    ax[1].hist(yr_te.to_numpy() - pred_r, bins=50, color=BLUE)
    ax[1].set_title("Distribution des residus"); ax[1].set_xlabel("residu = reel - predit (j)")
    fig.suptitle("Figure A.3 — Diagnostics de la regression de la duree (test scelle)",
                 fontsize=13, fontweight="bold")
    save(fig, "diagnostics.png")
except Exception as e:
    print("FAIL diagnostics.png:", e)

# ====================================== 6. learning_curve.png
try:
    pipe_lc = Pipeline([("pre", make_pre("continuous")),
                        ("m", HistGradientBoostingRegressor(random_state=RS,
                              **{k.replace("m__", ""): v for k, v in reg["best_params"].items()}))])
    Xlc, ylc = Xr_tr, yr_tr
    if len(Xr_tr) > 80000:
        Xlc, _, ylc, _ = train_test_split(Xr_tr, yr_tr, train_size=80000, random_state=RS)
    sizes, tr, va = learning_curve(clone(pipe_lc), Xlc, ylc, train_sizes=np.linspace(0.2, 1.0, 5),
                                   cv=3, scoring="r2", n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.plot(sizes, tr.mean(1), "o-", color=BLUE, label="apprentissage (train)")
    ax.plot(sizes, va.mean(1), "o-", color=ORANGE, label="validation")
    ax.set_xlabel("taille de l'echantillon d'apprentissage"); ax.set_ylabel("R2")
    ax.set_title("Figure A.4 — Courbe d'apprentissage (train vs validation)", fontsize=12, fontweight="bold")
    ax.legend()
    save(fig, "learning_curve.png")
except Exception as e:
    print("FAIL learning_curve.png:", e)

# ====================================== 7. meteo_ab.png (R2 sans vs avec meteo)
try:
    DAILY = ["temperature_2m_mean", "precipitation_sum", "windspeed_10m_max"]
    w = pd.read_parquet("ml_academique/weather_cache.parquet")
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    dfm = df.copy(); dfm["ship_day"] = dfm["ship_date"].dt.normalize()
    do = w.add_prefix("orig_").rename(columns={"orig_city": "origin_city", "orig_date": "ship_day"})
    dd = w.add_prefix("dest_").rename(columns={"dest_city": "destination_city", "dest_date": "ship_day"})
    dfm = dfm.merge(do, on=["origin_city", "ship_day"], how="left").merge(dd, on=["destination_city", "ship_day"], how="left")
    WN = [f"orig_{v}" for v in DAILY] + [f"dest_{v}" for v in DAILY]
    ym = dfm["realized_transit_days"].astype(float)
    params = {k.replace("m__", ""): v for k, v in reg["best_params"].items()}

    def fit_r2(num_cols):
        Xa = dfm[num_cols + OH + TE]
        Xtr, Xte, ya, yb = train_test_split(Xa, ym, test_size=0.20, random_state=RS)
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("oh", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("e", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), OH),
            ("te", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("e", TargetEncoder(target_type="continuous", cv=5, random_state=RS))]), TE)])
        p = Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=RS, **params))]).fit(Xtr, ya)
        return r2_score(yb, p.predict(Xte))

    r2_sans = fit_r2(NUM); r2_avec = fit_r2(NUM + WN)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    bars = ax.bar(["sans meteo", "avec meteo"], [r2_sans, r2_avec], color=[BLUE, GREEN], width=.55)
    for b, v in zip(bars, [r2_sans, r2_avec]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(0, max(r2_sans, r2_avec) * 1.15); ax.set_ylabel("R2 (test scelle)")
    ax.set_title(f"Figure A.5 — Apport de la meteo (gain DR2 = {r2_avec - r2_sans:+.4f})",
                 fontsize=12, fontweight="bold")
    save(fig, "meteo_ab.png")
    print(f"   meteo R2 sans={r2_sans:.4f} avec={r2_avec:.4f}")
except Exception as e:
    print("FAIL meteo_ab.png:", e)

# ====================================== 8. permutation.png (chute de R2)
try:
    perm = permutation_importance(reg["pipeline"], Xr_te, yr_te, scoring="r2",
                                  n_repeats=5, random_state=RS, n_jobs=-1)
    pim = pd.Series(perm.importances_mean, index=Xr_te.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(pim.index, pim.values, color=BLUE)
    ax.set_title("Figure A.6 — Importance par permutation (chute de R2 hors echantillon)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("chute de R2 quand la variable est permutee")
    save(fig, "permutation.png")
except Exception as e:
    print("FAIL permutation.png:", e)

# ====================================== 9. shap.png
try:
    import shap
    pre_r = reg["pipeline"].named_steps["pre"]; mdl_r = reg["pipeline"].named_steps["m"]
    Xe = Xr_te.sample(min(600, len(Xr_te)), random_state=RS)
    Xep = pre_r.transform(Xe); fn = pre_r.get_feature_names_out()
    expl = shap.Explainer(mdl_r, Xep)
    sv = expl(Xep)
    plt.figure()
    shap.summary_plot(sv, Xep, feature_names=fn, show=False, max_display=15)
    fig = plt.gcf(); fig.suptitle("Figure A.7 — Resume SHAP (regression de la duree)",
                                  fontsize=12, fontweight="bold", y=1.02)
    save(fig, "shap.png")
except Exception as e:
    print("FAIL shap.png (fallback importance):", e)
    try:
        sv_abs = np.abs(sv.values).mean(0)
        imp = pd.Series(sv_abs, index=fn).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(8.5, 5.5)); ax.barh(imp.index, imp.values, color=BLUE)
        ax.set_title("Figure A.7 — Importance SHAP moyenne |valeur|", fontsize=12, fontweight="bold")
        save(fig, "shap.png")
    except Exception as e2:
        print("  shap fallback failed:", e2)

# ====================================== 10. regle_risque_confusion.png
try:
    pred_r = reg["pipeline"].predict(Xr_te)
    est = Xr_te["estimated_transit_days"].to_numpy().astype(float)
    score = pred_r - est
    yhat = (score > 0).astype(int)
    y_exceed = (yr_te.to_numpy() > est).astype(int)
    cm = confusion_matrix(y_exceed, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-9)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["a l'heure", "depassement"], yticklabels=["a l'heure", "depassement"],
                annot_kws={"size": 13})
    ax.set_xlabel("predit (regle : duree predite > ETA)"); ax.set_ylabel("reel")
    ax.set_title(f"Figure A.8 — Regle de risque ETA\n(taux de fausses alertes = {far:.3f})",
                 fontsize=11, fontweight="bold")
    save(fig, "regle_risque_confusion.png")
    print(f"   regle risque tn/fp/fn/tp = {tn}/{fp}/{fn}/{tp}")
except Exception as e:
    print("FAIL regle_risque_confusion.png:", e)

# ---------------------------------------------------------------- recap
print("\n=== Fichiers crees dans ml_academique/figures/ ===")
order = ["eda.png", "clf_confusions.png", "clf_roc.png", "feature_selection.png",
         "diagnostics.png", "learning_curve.png", "meteo_ab.png", "permutation.png",
         "shap.png", "regle_risque_confusion.png"]
for nm in order:
    p = FIG / nm
    if p.exists():
        print(f"  {nm:30s} {p.stat().st_size/1024:7.1f} Ko   {'<-- CREE' if nm in created else '(deja present)'}")
    else:
        print(f"  {nm:30s}   MANQUANT")
