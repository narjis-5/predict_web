# -*- coding: utf-8 -*-
"""Genere delivery_duration_class_model_bundle.joblib + metriques clf dans
ml_academique/metadata.json. Reproduit EXACTEMENT la chaine de la section 17 du
notebook (cible classe de duree = realized > mediane ; pipeline anti-fuite ;
regression logistique ponderee). Aucune invention : metriques calculees sur le
test scelle."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

RS = 42
OUT = Path("ml_academique"); OUT.mkdir(exist_ok=True)
df = pd.read_parquet("nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet")
for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df = df.dropna(subset=["ship_date", "expected_delivery_date", "actual_delivery_date"]).copy()
df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days
df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)
df = df[(df["estimated_transit_days"] >= 0) & (df["realized_transit_days"] >= 0)].copy()

NUM = ["quantity", "shipping_cost_ngn", "estimated_transit_days"]
OH = ["supplier_name", "logistics_company", "ship_month", "ship_day_of_week"]
TE = ["origin_city", "destination_city"]
med = float(df["realized_transit_days"].median())
X = df[NUM + OH + TE].copy()
y = (df["realized_transit_days"] > med).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("oh", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("e", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), OH),
    ("te", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("e", TargetEncoder(target_type="binary", cv=5, random_state=RS))]), TE),
])
pipe = Pipeline([("pre", pre), ("m", LogisticRegression(class_weight="balanced",
                 solver="liblinear", max_iter=2000, random_state=RS))])
search = RandomizedSearchCV(pipe, {"m__C": [0.01, 0.1, 1.0, 10.0]}, n_iter=4, scoring="f1_macro",
                            cv=StratifiedKFold(4, shuffle=True, random_state=RS),
                            n_jobs=-1, random_state=RS, refit=False)
search.fit(Xtr, ytr)
best = Pipeline([("pre", pre), ("m", LogisticRegression(class_weight="balanced", solver="liblinear",
                 max_iter=2000, random_state=RS, **{k.replace("m__", ""): v for k, v in search.best_params_.items()}))]).fit(Xtr, ytr)

pr = best.predict(Xte); proba = best.predict_proba(Xte)[:, 1]
tn, fp, fn, tp = confusion_matrix(yte, pr, labels=[0, 1]).ravel()
metrics = {
    "clf_duree_target": "binaire_mediane", "clf_duree_seuil_jours": med,
    "clf_duree_best_model": "LogisticRegression", "clf_duree_best_params": search.best_params_,
    "clf_duree_accuracy": float(accuracy_score(yte, pr)),
    "clf_duree_bal_accuracy": float(balanced_accuracy_score(yte, pr)),
    "clf_duree_precision": float(precision_score(yte, pr, zero_division=0)),
    "clf_duree_recall": float(recall_score(yte, pr, zero_division=0)),
    "clf_duree_f1_macro": float(f1_score(yte, pr, average="macro")),
    "clf_duree_auc_roc": float(roc_auc_score(yte, proba)),
    "clf_duree_pr_auc": float(average_precision_score(yte, proba)),
    "clf_duree_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
bundle = {"task": "classification_classe_duree", "target": "binaire_mediane", "seuil_jours": med,
          "model_name": "LogisticRegression", "best_params": search.best_params_, "pipeline": best,
          "features": NUM + OH + TE,
          "feature_schema": {"numeric": NUM, "one_hot": OH, "target_encoded": TE}}
joblib.dump(bundle, "delivery_duration_class_model_bundle.joblib")

meta_path = OUT / "metadata.json"
meta = {}
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
meta.update(metrics)
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("bundle saved | seuil", round(med, 1),
      "| acc", round(metrics["clf_duree_accuracy"], 4),
      "| AUC", round(metrics["clf_duree_auc_roc"], 4),
      "| PR-AUC", round(metrics["clf_duree_pr_auc"], 4))
print("confusion tn/fp/fn/tp:", tn, fp, fn, tp)
