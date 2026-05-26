from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline
from gas_intelligence.config import (
    NLP_HIGH_CONFIDENCE_THRESHOLD,
    REMIT_LABEL_NAMES,
)
from gas_intelligence.nlp.preprocessing import clean_text


def classify_messages(
    pipe: Pipeline,
    texts: List[str],
    excerpt_length: int = 80,
    high_confidence_threshold: float = NLP_HIGH_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Classifie une liste de messages REMIT/UMM bruts.

    Parameters
    ----------
    texts : list[str]
        Messages tels que copiés depuis EEX, GRTgaz, ENTSOG, Gassco, etc.
        Le preprocessing (lowercase, strip ponctuation/chiffres) est appliqué
        automatiquement par le pipeline.

    Returns
    -------
    pd.DataFrame avec les colonnes :
        - message_excerpt : début du message (80 premiers caractères)
        - prediction     : "Planifié" ou "Incident"
        - proba_incident : probabilité estimée par le modèle (0 à 1)
        - confidence     : "haute" si proba > 0.85 ou < 0.15, sinon "moyenne"
    """
    if not texts:
        return pd.DataFrame(
            columns=["message_excerpt", "prediction", "proba_incident", "confidence"]
        )

    cleaned = [clean_text(t) for t in texts]
    probas = pipe.predict_proba(cleaned)[:, 1]
    preds = (probas >= 0.5).astype(int)

    low_threshold = 1.0 - high_confidence_threshold

    def _confidence(p: float) -> str:
        return "haute" if (p > high_confidence_threshold or p < low_threshold) else "moyenne"

    return pd.DataFrame(
        {
            "message_excerpt": [
                t[:excerpt_length] + ("…" if len(t) > excerpt_length else "")
                for t in texts
            ],
            "prediction": [REMIT_LABEL_NAMES[int(p)] for p in preds],
            "proba_incident": probas.round(3),
            "confidence": [_confidence(p) for p in probas],
        }
    )
