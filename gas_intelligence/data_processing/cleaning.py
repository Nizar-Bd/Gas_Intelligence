"""Nettoyage des profils de consommation : outliers, ratios infinis, contrôle qualité."""

import numpy as np
import pandas as pd

from gas_intelligence.config import (
    DEFAULT_OUTLIER_KEYS,
    INFINITE_RATIO_FILL,
    OUTLIER_CONSO_MEAN,
    OUTLIER_RATIO_HE,
    OUTLIER_VOLATILITY,
)


def identify_outliers(profils: pd.DataFrame, outlier_keys=None) -> list:
    """Retourne la liste des pointKeys à considérer comme outliers.

    Si ``outlier_keys`` est fourni, il est renvoyé tel quel. Sinon, la liste
    par défaut connue (``DEFAULT_OUTLIER_KEYS``) est utilisée, intersectée
    avec les pointKeys présents dans ``profils`` pour éviter les clés
    fantômes. En complément, on détecte aussi les points qui dépassent les
    seuils numériques définis dans ``config.py``.

    Parameters
    ----------
    profils : pd.DataFrame
        DataFrame contenant au moins la colonne ``pointKey``, et idéalement
        ``conso_mean``, ``volatilite`` et ``ratio_hiver_ete``.
    outlier_keys : list, optional
        Liste explicite de pointKeys à considérer comme outliers.

    Returns
    -------
    list[str]
        Liste de pointKeys identifiés comme outliers.
    """
    if outlier_keys is not None:
        return list(outlier_keys)

    known = [k for k in DEFAULT_OUTLIER_KEYS if k in profils["pointKey"].values]

    rule_based = []
    if {"conso_mean", "volatilite", "ratio_hiver_ete"}.issubset(profils.columns):
        mask = (
            (profils["conso_mean"] > OUTLIER_CONSO_MEAN)
            | (profils["volatilite"] > OUTLIER_VOLATILITY)
            | (profils["ratio_hiver_ete"] < OUTLIER_RATIO_HE)
        )
        rule_based = profils.loc[mask, "pointKey"].tolist()

    return sorted(set(known) | set(rule_based))


def remove_outliers(profils: pd.DataFrame, outlier_keys) -> pd.DataFrame:
    """Retire de ``profils`` les lignes dont ``pointKey`` figure dans ``outlier_keys``.

    Parameters
    ----------
    profils : pd.DataFrame
    outlier_keys : iterable[str]

    Returns
    -------
    pd.DataFrame
        Nouveau DataFrame sans les outliers (index réinitialisé).
    """
    cleaned = profils[~profils["pointKey"].isin(outlier_keys)].copy()
    return cleaned.reset_index(drop=True)


def encode_infinite_ratios(
    profils: pd.DataFrame,
    column: str = "ratio_hiver_ete",
    fill_value: float = INFINITE_RATIO_FILL,
) -> pd.DataFrame:
    """Remplace les valeurs infinies (``np.inf``, ``-np.inf``) de ``column`` par ``fill_value``.

    Utile pour les ratios saisonniers lorsque la consommation été est nulle
    (division par zéro → ``inf``). Le ``fill_value`` doit être choisi pour
    rester interprétable par les algorithmes en aval (clustering).

    Parameters
    ----------
    profils : pd.DataFrame
    column : str, default ``"ratio_hiver_ete"``
    fill_value : float, default ``INFINITE_RATIO_FILL``

    Returns
    -------
    pd.DataFrame
        Copie de ``profils`` avec les ``inf`` remplacés.
    """
    out = profils.copy()
    out[column] = out[column].replace([np.inf, -np.inf], fill_value)
    return out


def validate_data_quality(profils: pd.DataFrame) -> None:
    """Affiche un rapport de qualité des données (NaN, ranges, types).

    Imprime sur stdout :
        - le nombre de lignes / colonnes,
        - le pourcentage de NaN par colonne,
        - pour chaque colonne numérique : min, max, mean, et nombre de
          valeurs non finies (inf, -inf).
    """
    n = len(profils)
    print("=" * 60)
    print(f"Rapport de qualite — {n} lignes, {profils.shape[1]} colonnes")
    print("=" * 60)

    print("\nValeurs manquantes :")
    nan_counts = profils.isna().sum()
    for col, count in nan_counts.items():
        if count > 0:
            pct = 100 * count / n if n else 0
            print(f"  - {col:20s} : {count:3d} NaN ({pct:5.1f}%)")
    if nan_counts.sum() == 0:
        print("  Aucune valeur manquante.")

    numeric_cols = profils.select_dtypes(include="number").columns
    print("\nStatistiques par feature numerique :")
    for col in numeric_cols:
        s = profils[col]
        n_inf = int((~np.isfinite(s.dropna())).sum())
        print(
            f"  - {col:20s} | min={s.min():10.3f}  max={s.max():12.3f}  "
            f"mean={s.mean():10.3f}  inf={n_inf}"
        )
    print("=" * 60)
