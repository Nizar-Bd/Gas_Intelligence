from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from gas_intelligence.config import (
    LOGREG_C,
    LOGREG_MAX_ITER,
    NLP_RANDOM_STATE,
    REMIT_LABEL_NAMES,
    TFIDF_LOWERCASE,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)


def build_classifier_pipeline(
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    C: float = LOGREG_C,
    max_iter: int = LOGREG_MAX_ITER,
    random_state: int = NLP_RANDOM_STATE,
) -> Pipeline:
    """Construit le pipeline ``TfidfVectorizer → LogisticRegression``.

    Les hyperparamètres par défaut sont ceux validés dans le notebook 2 et
    centralisés dans ``gas_intelligence.config``.

    Parameters
    ----------
    ngram_range : tuple, default ``TFIDF_NGRAM_RANGE``
        Plage de n-grammes capturés par le ``TfidfVectorizer``.
    min_df : int, default ``TFIDF_MIN_DF``
        Fréquence document minimale pour garder un token.
    C : float, default ``LOGREG_C``
        Inverse de la force de régularisation L2.
    max_iter : int, default ``LOGREG_MAX_ITER``
        Itérations max du solveur de la régression logistique.
    random_state : int, default ``NLP_RANDOM_STATE``
        Seed pour la reproductibilité.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline non entraîné, prêt à recevoir un ``.fit(X, y)``.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    stop_words=TFIDF_STOP_WORDS,
                    lowercase=TFIDF_LOWERCASE,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_classifier(X_train, y_train, **build_kwargs) -> Pipeline:
    """Construit et entraîne le pipeline en une seule étape.

    Wrapper de convenance autour de ``build_classifier_pipeline().fit(...)``.

    Parameters
    ----------
    X_train : iterable[str]
        Messages nettoyés (la sortie de ``clean_text`` appliquée à chaque ligne).
    y_train : iterable[int]
        Labels associés (0 = planifié, 1 = incident).
    **build_kwargs
        Paramètres optionnels passés à ``build_classifier_pipeline``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline entraîné sur ``(X_train, y_train)``.
    """
    pipe = build_classifier_pipeline(**build_kwargs)
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_classifier(pipe: Pipeline, X_test, y_test) -> Dict[str, Any]:
    """Évalue un pipeline entraîné sur un jeu de test.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline déjà entraîné par ``train_classifier`` (ou ``.fit`` manuel).
    X_test : iterable[str]
        Messages de test nettoyés.
    y_test : iterable[int]
        Labels de référence.

    Returns
    -------
    dict
        Dictionnaire contenant :
            - ``accuracy``         : float — accuracy globale
            - ``report``           : str   — sortie de ``classification_report``
            - ``confusion_matrix`` : np.ndarray — matrice 2x2 (lignes = vérité)
            - ``y_pred``           : np.ndarray — prédictions (utile pour l'inspection
                                                 des erreurs côté notebook)
    """
    y_pred = pipe.predict(X_test)
    target_names = [
        f"{REMIT_LABEL_NAMES[0]} (0)",
        f"{REMIT_LABEL_NAMES[1]} (1)",
    ]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, target_names=target_names),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


def extract_feature_importance(pipe: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extrait les ``top_n`` tokens les plus discriminants par direction.

    Pour une ``LogisticRegression`` binaire avec classe positive = 1 (incident) :
        - coefficient positif → pousse vers **incident**,
        - coefficient négatif → pousse vers **planifié**.

    Le DataFrame retourné est trié de manière à pouvoir être passé directement
    à ``matplotlib.barh`` (les tokens "planifié" sont en haut, du plus négatif
    au moins négatif ; les tokens "incident" en bas, du moins positif au plus
    positif), tout en gardant la colonne ``direction`` pour filtrer.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline entraîné contenant un step ``"tfidf"`` et un step ``"clf"``.
    top_n : int, default 15
        Nombre de tokens à extraire de chaque côté (incident / planifié).

    Returns
    -------
    pd.DataFrame
        Colonnes :
            - ``token``     : str   — n-gramme appris par le vectoriseur
            - ``coef``      : float — coefficient de la régression logistique
            - ``direction`` : str   — ``"→ INCIDENT"`` ou ``"→ PLANIFIÉ"``
        Longueur = ``2 * top_n``.
    """
    vectorizer: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    importance = pd.DataFrame({"token": feature_names, "coef": coefs})

    top_incident = (
        importance.nlargest(top_n, "coef")
        .assign(direction="→ INCIDENT")
        .sort_values("coef", ascending=False)
    )
    top_planned = (
        importance.nsmallest(top_n, "coef")
        .assign(direction="→ PLANIFIÉ")
        .sort_values("coef")
    )

    return pd.concat([top_incident, top_planned], ignore_index=True)
