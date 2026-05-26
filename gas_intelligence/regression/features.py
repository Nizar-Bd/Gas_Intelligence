from typing import Tuple

import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out["jour_semaine"]   = out[date_col].dt.dayofweek
    out["mois"]           = out[date_col].dt.month
    out["annee"]          = out[date_col].dt.year
    out["trimestre"]      = out[date_col].dt.quarter
    out["est_weekend"]    = (out["jour_semaine"] >= 5).astype(int)
    out["est_lundi"]      = (out["jour_semaine"] == 0).astype(int)
    out["num_jour_annee"] = out[date_col].dt.dayofyear
    return out


def add_thermal_features(
    df: pd.DataFrame,
    temp_cols: Tuple[str, ...] = (
        "temp_amsterdam_celsius",
        "temp_berlin_celsius",
        "temp_paris_celsius",
    ),
    base_temp: float = 15.0,
) -> pd.DataFrame:
    # HDD = max(0, base_temp - T_moy), proxy demande chauffage
    out = df.copy()
    out["temp_moyenne_eur"] = out[list(temp_cols)].mean(axis=1)
    out["hdd"] = (base_temp - out["temp_moyenne_eur"]).clip(lower=0)
    return out


def add_ratio_features(df: pd.DataFrame, mmbtu_to_mwh: float = 3.6) -> pd.DataFrame:
    # Spreads cross-marches, conversion MMBtu vers MWh sur Henry Hub
    out = df.copy()
    out["ratio_brent_ttf"]     = out["prix_brent_usd_per_bbl"] / out["prix_ttf_eur_per_mwh"]
    out["ratio_ttf_henry_hub"] = out["prix_ttf_eur_per_mwh"]   / (out["prix_henry_hub_usd_per_mmbtu"] * mmbtu_to_mwh)
    out["ratio_jkm_henry_hub"] = out["prix_jkm_usd_per_mmbtu"] / out["prix_henry_hub_usd_per_mmbtu"]
    return out


def add_volatility_feature(
    df: pd.DataFrame,
    price_col: str = "prix_ttf_eur_per_mwh",
    window: int = 14,
) -> pd.DataFrame:
    # ATR simplifie close-only (vrai ATR demande OHLC)
    out = df.copy()
    out["volatilite_atr14"] = out[price_col].diff().abs().rolling(window=window).mean()
    return out
