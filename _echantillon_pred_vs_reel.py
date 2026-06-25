# -*- coding: utf-8 -*-
"""Échantillon 15 expéditions du TEST : durée réelle vs prédite par le modèle.
Reproduit EXACTEMENT la partition du notebook (engineer/split identiques) puis
applique le pipeline du bundle. Rien n'est inventé : tout vient du parquet et du modèle.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(r"C:\Users\user\Desktop\pfeN")
RANDOM_STATE = 42
TEST_SIZE = 0.20
NUM_BASE = ["quantity", "shipping_cost_ngn", "estimated_transit_days"]
OH = ["supplier_name", "logistics_company", "ship_month", "ship_day_of_week"]
TE = ["origin_city", "destination_city"]

# --- 1. Charger le bundle et le parquet -------------------------------------
bundle = joblib.load(HERE / "delivery_duration_model_bundle.joblib")
pipe = bundle["pipeline"]
df_raw = pd.read_parquet(HERE / "nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet")

# --- 2. Feature engineering + filtrage IDENTIQUES au notebook ---------------
df = df_raw.copy()
for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df = df.dropna(subset=["ship_date", "expected_delivery_date", "actual_delivery_date"]).copy()

df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)
df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days

df = df[(df["estimated_transit_days"] >= 0) & (df["realized_transit_days"] >= 0)].copy()
df = df.dropna(subset=["estimated_transit_days", "realized_transit_days"]).copy()

# --- 3. Même partition train/test (mêmes indices via random_state) ----------
X_all = df[NUM_BASE + OH + TE].copy()
y_reg = df["realized_transit_days"].astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Test set: {X_te.shape[0]} expéditions")

# --- 4. Tirage de 15 expéditions de TEST au hasard (random_state=42) --------
ech_idx = X_te.sample(15, random_state=RANDOM_STATE).index

# --- 5. Durée réelle (depuis les dates) et durée prédite (par le modèle) -----
duree_reelle = (df.loc[ech_idx, "actual_delivery_date"] - df.loc[ech_idx, "ship_date"]).dt.days
duree_predite = pipe.predict(X_all.loc[ech_idx])

out = pd.DataFrame({
    "origin_city": df.loc[ech_idx, "origin_city"].values,
    "destination_city": df.loc[ech_idx, "destination_city"].values,
    "logistics_company": df.loc[ech_idx, "logistics_company"].values,
    "duree_reelle_j": duree_reelle.values,
    "duree_predite_j": duree_predite.round(1),
})
out["erreur_j"] = (out["duree_predite_j"] - out["duree_reelle_j"]).round(1)
out["duree_predite_j"] = out["duree_predite_j"].round(1)

# --- 6. Affichage + export --------------------------------------------------
pd.set_option("display.width", 160, "display.max_columns", None)
print()
print(out.to_string(index=False))

mae_ech = out["erreur_j"].abs().mean()
print(f"\nErreur absolue moyenne de l'échantillon (MAE) : {mae_ech:.2f} j")

dest = HERE / "ml_academique" / "tables" / "echantillon_pred_vs_reel.csv"
out.to_csv(dest, index=False, encoding="utf-8")
print(f"Exporté -> {dest}")
