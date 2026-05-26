"""Construction des 4 features utilisées pour le clustering."""

import numpy as np
import pandas as pd

from gas_intelligence.config import (
    SUMMER_MONTHS,
    WEEKEND_DAYS,
    WINTER_MONTHS,
)


def calculate_consumption_mean(daily_data: pd.DataFrame, value_column: str = "value") -> float:
    """Moyenne de la consommation quotidienne sur la période.

    Parameters
    ----------
    daily_data : pd.DataFrame
        DataFrame avec une ligne par jour.
    value_column : str, default ``"value"``

    Returns
    -------
    float
    """
    return float(daily_data[value_column].mean())


def calculate_volatility(daily_data: pd.DataFrame, value_column: str = "value") -> float:
    """Coefficient de variation σ/μ de la consommation.

    Retourne ``np.nan`` si la moyenne est nulle (division par zéro).
    """
    mean = daily_data[value_column].mean()
    if mean == 0 or pd.isna(mean):
        return np.nan
    std = daily_data[value_column].std()
    return float(std / mean)


def calculate_seasonal_ratio(
    daily_data: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "value",
) -> float:
    """Ratio conso_hiver / conso_été.

    Hiver = mois ``WINTER_MONTHS`` (jan, fev, mar, oct, nov, dec).
    Été   = mois ``SUMMER_MONTHS`` (juin, juil, août).

    Retourne ``np.inf`` si la moyenne été est nulle (et hiver > 0),
    et ``np.nan`` si aucune des deux saisons n'a de données.
    """
    dates = pd.to_datetime(daily_data[date_column])
    months = dates.dt.month
    winter_mask = months.isin(WINTER_MONTHS)
    summer_mask = months.isin(SUMMER_MONTHS)

    winter_mean = daily_data.loc[winter_mask, value_column].mean()
    summer_mean = daily_data.loc[summer_mask, value_column].mean()

    if pd.isna(winter_mean) and pd.isna(summer_mean):
        return np.nan
    if pd.isna(summer_mean) or summer_mean == 0:
        # Pas de conso d'été : ratio infini par convention
        return np.inf if (winter_mean and winter_mean > 0) else np.nan
    if pd.isna(winter_mean):
        return np.nan
    return float(winter_mean / summer_mean)


def calculate_weekend_weekday_ratio(
    daily_data: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "value",
) -> float:
    """Ratio conso_weekend / conso_weekday.

    Weekend = jours ``WEEKEND_DAYS`` (samedi=5, dimanche=6).

    Retourne ``np.nan`` si l'une des deux moyennes manque ou si la moyenne
    des jours de semaine est nulle.
    """
    dates = pd.to_datetime(daily_data[date_column])
    dow = dates.dt.dayofweek
    weekend_mask = dow.isin(WEEKEND_DAYS)

    weekend_mean = daily_data.loc[weekend_mask, value_column].mean()
    weekday_mean = daily_data.loc[~weekend_mask, value_column].mean()

    if pd.isna(weekend_mean) or pd.isna(weekday_mean) or weekday_mean == 0:
        return np.nan
    return float(weekend_mean / weekday_mean)


def engineer_all_features(
    raw_data: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "value",
    group_by: str = "pointKey",
) -> pd.DataFrame:
    """Construit les 4 features de clustering pour chaque entité (``group_by``).

    Parameters
    ----------
    raw_data : pd.DataFrame
        Données quotidiennes brutes — doit contenir ``group_by``,
        ``date_column`` et ``value_column``.
    date_column : str, default ``"date"``
    value_column : str, default ``"value"``
    group_by : str, default ``"pointKey"``

    Returns
    -------
    pd.DataFrame
        Une ligne par entité ``group_by`` avec les colonnes :
        ``conso_mean``, ``volatilite``, ``ratio_hiver_ete``, ``ratio_we_wd``.
    """
    rows = []
    for key, sub in raw_data.groupby(group_by):
        rows.append(
            {
                group_by: key,
                "conso_mean": calculate_consumption_mean(sub, value_column=value_column),
                "volatilite": calculate_volatility(sub, value_column=value_column),
                "ratio_hiver_ete": calculate_seasonal_ratio(
                    sub, date_column=date_column, value_column=value_column
                ),
                "ratio_we_wd": calculate_weekend_weekday_ratio(
                    sub, date_column=date_column, value_column=value_column
                ),
            }
        )
    return pd.DataFrame(rows)
