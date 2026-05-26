"""Preprocessing texte et chargement du dataset REMIT/UMM."""

import re
from pathlib import Path

import pandas as pd

from gas_intelligence.config import DEFAULT_REMIT_CSV


def clean_text(text: str) -> str:
    """Normalise un message UMM brut en une chaîne lisible par le TF-IDF.

    Les opérations appliquées sont volontairement minimales :
        - passage en minuscules ;
        - suppression de tout caractère non alphabétique (chiffres, ponctuation,
          symboles), qui ne discrimine pas planifié vs incident ;
        - collapse des espaces multiples.

    On ne fait ni stemming, ni lemmatisation, ni stop-word removal (cf. les
    consignes du projet : les bigrammes type ``due to`` peuvent porter du
    signal sur des messages aussi courts).

    Parameters
    ----------
    text : str
        Message brut tel que publié par un TSO (EEX, ENTSOG, GRTgaz, etc.).

    Returns
    -------
    str
        Chaîne nettoyée prête à être passée au ``TfidfVectorizer``.
    """
    text = text.lower()
    # remplace tout ce qui n'est pas une lettre par un espace
    text = re.sub(r"[^a-z]+", " ", text)
    # collapse multi-espaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_remit_dataset(csv_path: str = DEFAULT_REMIT_CSV) -> pd.DataFrame:
    """Charge le faux dataset REMIT et garantit le typage des colonnes.

    Si ``csv_path`` est un chemin relatif, il est résolu **depuis la racine du
    projet** (parent de ``gas_intelligence/``). On évite ainsi que la fonction
    se casse selon le répertoire d'exécution (notebook vs script vs test).

    Parameters
    ----------
    csv_path : str, default ``DEFAULT_REMIT_CSV``
        Chemin vers le CSV à deux colonnes ``message,label``.

    Returns
    -------
    pd.DataFrame
        DataFrame avec deux colonnes :
            - ``message`` : str — le message brut
            - ``label``   : int — 0 = planifié, 1 = incident
    """
    path = Path(csv_path)
    if not path.is_absolute():
        # gas_intelligence/nlp/preprocessing.py → parent.parent.parent = racine du projet
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / path

    df = pd.read_csv(path)
    df["message"] = df["message"].astype(str)
    df["label"] = df["label"].astype(int)
    return df
