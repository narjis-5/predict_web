# -*- coding: utf-8 -*-
"""
train_model.py
==============
Régénère le modèle de déploiement DANS l'environnement du site, ce qui évite
tout problème de version (un modèle picklé sous une version de scikit-learn et
relu sous une autre échoue). À lancer UNE fois :

    python train_model.py

Prérequis : le fichier parquet dans le même dossier. Internet (météo) recommandé
mais non bloquant. Produit :
    - delivery_duration_model_bundle.joblib   (pipeline + schéma + règle de décision)
    - model_report.json                        (métriques + interprétation pour le site)
    - weather_cache.parquet                    (cache météo, réutilisé par le site)

Hyperparamètres : ceux retenus par l'optimisation du notebook (RandomizedSearchCV).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import pipeline_utils as pu

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_FILE = "nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "weather_cache.parquet"

# Hyperparamètres optimisés (résultat de la recherche aléatoire du notebook)
BEST_PARAMS = dict(learning_rate=0.2, max_iter=200, max_leaf_nodes=31,
                   min_samples_leaf=100, l2_regularization=5.0, random_state=RANDOM_STATE)


def make_preprocessor(num_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("oh",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("e", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), pu.OH),
        ("te",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("e", TargetEncoder(target_type="continuous", cv=5, random_state=RANDOM_STATE))]), pu.TE),
    ], remainder="drop")


def regression_metrics(y_true, pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(mean_squared_error(y_true, pred) ** 0.5),
        "R2": float(r2_score(y_true, pred)),
    }


def risk_rule_metrics(y_true_duration, estimated_duration, predicted_duration) -> dict:
    """Evaluate the deployment rule as a diagnostic classifier.

    The threshold optimized on the test split is reported only as a diagnostic
    sensitivity analysis; it must not be treated as a production-tuned threshold.
    """
    estimated = np.asarray(estimated_duration, dtype=float)
    score = np.asarray(predicted_duration, dtype=float) - estimated
    y_exceed = (np.asarray(y_true_duration, dtype=float) > estimated).astype(int)
    yhat0 = (score > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_exceed, yhat0, labels=[0, 1]).ravel()

    prec, rec, thr = precision_recall_curve(y_exceed, score)
    if len(thr):
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        best_i = int(np.argmax(f1s))
        diagnostic_threshold = float(thr[best_i])
        yhat_diag = (score >= diagnostic_threshold).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(y_exceed, yhat_diag, labels=[0, 1]).ravel()
    else:
        diagnostic_threshold = 0.0
        yhat_diag = yhat0
        tn2, fp2, fn2, tp2 = tn, fp, fn, tp

    return {
        "base_rate": float(y_exceed.mean()),
        "default_threshold_days": 0.0,
        "default_precision": float(precision_score(y_exceed, yhat0, zero_division=0)),
        "default_recall": float(recall_score(y_exceed, yhat0, zero_division=0)),
        "default_f1": float(f1_score(y_exceed, yhat0, zero_division=0)),
        "default_false_alarm_rate": float(fp / (fp + tn + 1e-9)),
        "default_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc": float(roc_auc_score(y_exceed, score)),
        "pr_auc": float(average_precision_score(y_exceed, score)),
        "diagnostic_best_f1_threshold_days": diagnostic_threshold,
        "diagnostic_best_f1_precision": float(precision_score(y_exceed, yhat_diag, zero_division=0)),
        "diagnostic_best_f1_recall": float(recall_score(y_exceed, yhat_diag, zero_division=0)),
        "diagnostic_best_f1": float(f1_score(y_exceed, yhat_diag, zero_division=0)),
        "diagnostic_best_f1_false_alarm_rate": float(fp2 / (fp2 + tn2 + 1e-9)),
        "diagnostic_best_f1_confusion_matrix": {"tn": int(tn2), "fp": int(fp2), "fn": int(fn2), "tp": int(tp2)},
        "interpretation": (
            "Diagnostic only: a high F1 can be inflated by the high base rate. "
            "ROC-AUC near 0.5 means the rule does not discriminate individual ETA exceedances."
        ),
    }


def main():
    data_path = HERE / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Place {DATA_FILE} dans {HERE}")

    print("Lecture des données…")
    df_raw = pd.read_parquet(data_path)
    df, _ = pu.engineer_features(df_raw)
    df = df[df["realized_transit_days"].notna() & (df["realized_transit_days"] >= 0)].copy()

    print("Récupération de la météo (groupée par ville, mise en cache)…")
    cities = sorted(set(df["origin_city"]) | set(df["destination_city"]))
    start = df["ship_date"].min().strftime("%Y-%m-%d")
    end = df["ship_date"].max().strftime("%Y-%m-%d")
    weather = pu.fetch_weather(cities, start, end, cache_path=CACHE)
    df, weather_num = pu.merge_weather(df, weather)
    weather_ok = weather is not None
    print("Météo incluse :", weather_ok)

    NUM = pu.NUM_BASE + (weather_num if weather_ok else [])
    X = pu.build_X(df, weather_num if weather_ok else [])
    y = df["realized_transit_days"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("Entraînement du modèle (HistGradientBoosting, hyperparamètres optimisés)…")
    pipe = Pipeline([("pre", make_preprocessor(NUM)),
                     ("m", HistGradientBoostingRegressor(**BEST_PARAMS))]).fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    metrics = regression_metrics(y_te, pred)
    mae = metrics["MAE"]
    rmse = metrics["RMSE"]
    r2 = metrics["R2"]
    naive_r2 = float(r2_score(y_te, X_te["estimated_transit_days"]))

    print("Validation temporelle passe -> futur...")
    dft = df.sort_values("ship_date").reset_index(drop=True)
    cut = int(len(dft) * 0.80)
    df_time_tr = dft.iloc[:cut].copy()
    df_time_te = dft.iloc[cut:].copy()
    X_time_tr = pu.build_X(df_time_tr, weather_num if weather_ok else [])
    X_time_te = pu.build_X(df_time_te, weather_num if weather_ok else [])
    y_time_tr = df_time_tr["realized_transit_days"].astype(float)
    y_time_te = df_time_te["realized_transit_days"].astype(float)
    pipe_time = Pipeline([("pre", make_preprocessor(NUM)),
                          ("m", HistGradientBoostingRegressor(**BEST_PARAMS))]).fit(X_time_tr, y_time_tr)
    pred_time = pipe_time.predict(X_time_te)
    temporal_metrics = regression_metrics(y_time_te, pred_time)
    temporal_metrics.update({
        "naive_R2": float(r2_score(y_time_te, X_time_te["estimated_transit_days"])),
        "train_start": df_time_tr["ship_date"].min().strftime("%Y-%m-%d"),
        "train_end": df_time_tr["ship_date"].max().strftime("%Y-%m-%d"),
        "test_start": df_time_te["ship_date"].min().strftime("%Y-%m-%d"),
        "test_end": df_time_te["ship_date"].max().strftime("%Y-%m-%d"),
        "n_train": int(len(X_time_tr)),
        "n_test": int(len(X_time_te)),
    })
    risk_metrics = risk_rule_metrics(y_te, X_te["estimated_transit_days"], pred)

    # Contre-épreuve : le retard seul
    delay_r2 = None
    if "delay_days" in df.columns:
        yd = df["delay_days"].astype(float)
        Xtr2, Xte2, ytr2, yte2 = train_test_split(X, yd, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        pipe_d = Pipeline([("pre", make_preprocessor(NUM)),
                           ("m", HistGradientBoostingRegressor(**BEST_PARAMS))]).fit(Xtr2, ytr2)
        delay_r2 = float(r2_score(yte2, pipe_d.predict(Xte2)))

    print("Calcul de l'importance par permutation…")
    Xperm = X_te.sample(min(20000, len(X_te)), random_state=RANDOM_STATE)
    yperm = y_te.loc[Xperm.index]
    perm = permutation_importance(pipe, Xperm, yperm, scoring="r2",
                                  n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"variable": X_te.columns, "chute_R2": perm.importances_mean})
                  .sort_values("chute_R2", ascending=False))

    bundle = {
        "task": "regression_duree_livraison",
        "pipeline": pipe,
        "model_name": "HistGradientBoosting",
        "best_params": BEST_PARAMS,
        "weather_enabled": bool(weather_ok),
        "weather_features": weather_num,
        "weather_daily_vars": pu.DAILY_VARS,
        "feature_schema": {"numeric": NUM, "one_hot": pu.OH, "target_encoded": pu.TE},
        "decision_rule": "risque_depassement_ETA = (duree_predite > estimated_transit_days)",
        "decision_rule_warning": (
            "Operational prioritization indicator, not a calibrated probability "
            "of individual ETA overrun."
        ),
        "mae_days": mae,
    }
    joblib.dump(bundle, HERE / "delivery_duration_model_bundle.joblib")

    report = {
        "model": "HistGradientBoosting", "best_params": BEST_PARAMS,
        "weather_enabled": bool(weather_ok),
        "test_MAE": mae, "test_RMSE": rmse, "test_R2": r2,
        "naive_R2": naive_r2, "delay_R2": delay_r2,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "temporal_validation": temporal_metrics,
        "risk_rule_metrics": risk_metrics,
        "scientific_conclusion": (
            "Les donnees disponibles permettent surtout de recalibrer l'ETA. "
            "La regle de risque doit etre lue comme un indicateur operationnel "
            "de priorisation, pas comme un detecteur fiable de retards individuels."
        ),
        "permutation_importance": importance.to_dict(orient="records"),
    }
    json.dump(report, open(HERE / "model_report.json", "w"), ensure_ascii=False, indent=2)

    print("\n=== Terminé ===")
    print(f"R2={r2:.3f} | MAE={mae:.3f} j | naïf R2={naive_r2:.3f} | retard R2={delay_r2}")
    print("Fichiers écrits : delivery_duration_model_bundle.joblib, model_report.json, weather_cache.parquet")

    print(f"Temporel R2={temporal_metrics['R2']:.3f} | AUC risque={risk_metrics['roc_auc']:.3f} | fausses alertes={risk_metrics['default_false_alarm_rate']:.3f}")


if __name__ == "__main__":
    main()
