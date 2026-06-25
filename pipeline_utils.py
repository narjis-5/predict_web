# -*- coding: utf-8 -*-
"""
pipeline_utils.py
=================
Module partagé par `train_model.py` (entraînement) et `app.py` (déploiement).
Le fait que les DEUX importent ces fonctions garantit que le site applique
EXACTEMENT les mêmes étapes que le notebook : même feature engineering, même
enrichissement météo, même schéma de variables. Aucune divergence possible.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SCHÉMA — identique au notebook final
# ---------------------------------------------------------------------------
NUM_BASE = ["quantity", "shipping_cost_ngn", "estimated_transit_days"]
OH = ["supplier_name", "logistics_company", "ship_month", "ship_day_of_week"]
TE = ["origin_city", "destination_city"]
DAILY_VARS = ["temperature_2m_mean", "precipitation_sum", "windspeed_10m_max"]
TARGET = "realized_transit_days"

# Colonnes brutes minimales pour produire une prédiction
REQUIRED_RAW = [
    "ship_date", "expected_delivery_date", "origin_city", "destination_city",
    "logistics_company", "supplier_name", "quantity", "shipping_cost_ngn",
]

# Coordonnées des villes nigérianes (repli géocodage Open-Meteo si inconnue)
CITY_COORDS = {
    "Lagos": (6.5244, 3.3792), "Abuja": (9.0765, 7.3986), "Kano": (12.0022, 8.5920),
    "Port Harcourt": (4.8156, 7.0498), "Warri": (5.5167, 5.7500), "Aba": (5.1066, 7.3667),
    "Ibadan": (7.3775, 3.9470), "Jos": (9.8965, 8.8583), "Enugu": (6.4584, 7.5464),
    "Kaduna": (10.5222, 7.4383), "Onitsha": (6.1667, 6.7833), "Abeokuta": (7.1557, 3.3451),
    "Benin City": (6.3350, 5.6037), "Calabar": (4.9589, 8.3269), "Ilorin": (8.4966, 4.5421),
}


# ---------------------------------------------------------------------------
# 1. LECTURE MULTI-FORMAT
# ---------------------------------------------------------------------------
def read_any(file_like, name: str) -> pd.DataFrame:
    """Lit csv / txt / xlsx / xls / parquet / json sans présumer du format."""
    n = name.lower()
    if hasattr(file_like, "read"):
        raw = file_like.read()
        buf = io.BytesIO(raw)
    else:  # chemin disque
        buf = file_like
    if n.endswith((".csv", ".txt")):
        for sep in [",", ";", "\t", "|"]:
            try:
                if hasattr(buf, "seek"):
                    buf.seek(0)
                df = pd.read_csv(buf, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        if hasattr(buf, "seek"):
            buf.seek(0)
        return pd.read_csv(buf)
    if n.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    if n.endswith(".parquet"):
        return pd.read_parquet(buf)
    if n.endswith(".json"):
        return pd.read_json(buf)
    raise ValueError("Format non reconnu : utilisez csv, xlsx, parquet ou json.")


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING — anti-fuite, identique au notebook
# ---------------------------------------------------------------------------
def engineer_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construit les variables disponibles avant l'expédition + cibles si possible."""
    rapport = {"lignes_initiales": int(len(df_raw))}
    df = df_raw.copy()

    for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
    df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
    df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)

    has_actual = "actual_delivery_date" in df.columns and df["actual_delivery_date"].notna().any()
    if has_actual:
        df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days
        df["delay_days"] = (df["actual_delivery_date"] - df["expected_delivery_date"]).dt.days.clip(lower=0)

    before = len(df)
    df = df.dropna(subset=["estimated_transit_days", "ship_month", "ship_day_of_week"]).copy()
    df = df[df["estimated_transit_days"] >= 0].copy()
    rapport["lignes_incoherentes_retirees"] = int(before - len(df))
    rapport["lignes_finales"] = int(len(df))
    rapport["a_realized"] = bool(has_actual)
    return df, rapport


# ---------------------------------------------------------------------------
# 3. MÉTÉO — appels groupés par ville, réessais, cache, repli gracieux
# ---------------------------------------------------------------------------
def _geocode(name: str):
    try:
        import requests
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name": name, "count": 1, "country": "NG"}, timeout=30)
        res = r.json().get("results")
        if res:
            return (res[0]["latitude"], res[0]["longitude"])
    except Exception:
        return None
    return None


def _fetch_city(name, lat, lon, start, end, retries=3):
    import requests
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "daily": ",".join(DAILY_VARS), "timezone": "Africa/Lagos"}
    last = None
    for k in range(retries):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params=params, timeout=60)
            r.raise_for_status()
            daily = r.json()["daily"]
            w = pd.DataFrame(daily).rename(columns={"time": "date"})
            w["date"] = pd.to_datetime(w["date"])
            w["city"] = name
            return w
        except Exception as e:
            last = e
            time.sleep(2 * (k + 1))
    return None


def fetch_weather(cities, start, end, cache_path: Path | None = None, log=print):
    """Renvoie un DataFrame météo (city, date, variables) ou None si tout échoue."""
    if cache_path is not None and Path(cache_path).exists():
        try:
            cached = pd.read_parquet(cache_path)
            have = set(cached["city"].unique())
            if set(cities).issubset(have):
                log(f"Météo lue depuis le cache ({len(cached)} lignes).")
                return cached
        except Exception:
            pass

    frames = []
    for c in cities:
        coords = CITY_COORDS.get(c) or _geocode(c)
        if coords is None:
            log(f"Coordonnées introuvables pour {c}, ville ignorée.")
            continue
        w = _fetch_city(c, coords[0], coords[1], start, end)
        if w is not None:
            frames.append(w)
        time.sleep(0.3)
    if not frames:
        return None
    weather = pd.concat(frames, ignore_index=True)
    if cache_path is not None:
        try:
            weather.to_parquet(cache_path, index=False)
        except Exception:
            pass
    return weather


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame | None) -> tuple[pd.DataFrame, list]:
    """Jointure sans fuite : météo origine + destination à la DATE D'EXPÉDITION.

    Crée toujours les 6 colonnes météo (NaN si indisponibles) pour rester
    compatible avec le schéma du modèle entraîné.
    """
    weather_num = [f"orig_{v}" for v in DAILY_VARS] + [f"dest_{v}" for v in DAILY_VARS]
    df = df.copy()
    df["ship_day"] = df["ship_date"].dt.normalize()

    if weather is not None and len(weather):
        w = weather.copy()
        w["date"] = pd.to_datetime(w["date"]).dt.normalize()
        do = w.add_prefix("orig_").rename(columns={"orig_city": "origin_city", "orig_date": "ship_day"})
        dd = w.add_prefix("dest_").rename(columns={"dest_city": "destination_city", "dest_date": "ship_day"})
        df = df.merge(do, on=["origin_city", "ship_day"], how="left")
        df = df.merge(dd, on=["destination_city", "ship_day"], how="left")
    # Garantir la présence des colonnes (le modèle les attend ; imputation médiane en aval)
    for col in weather_num:
        if col not in df.columns:
            df[col] = np.nan
    return df, weather_num


# ---------------------------------------------------------------------------
# 4. MATRICE DE VARIABLES (ordre indifférent : le ColumnTransformer cible par nom)
# ---------------------------------------------------------------------------
def build_X(df: pd.DataFrame, weather_num: list) -> pd.DataFrame:
    cols = NUM_BASE + weather_num + OH + TE
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols].copy()
