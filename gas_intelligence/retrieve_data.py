import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_consumption_data(
    date_start: str = None,
    date_end: str = None,
    point_type_id: int = 18,  # 18 = Aggregated Point - Final Consumers
    period_type: str = "day",
    timezone: str = "CET",
    output_dir: str = "data/raw",
    chunk_days: int = 180,
    timeout: int = 180,
    max_retries: int = 3,
    retry_wait: int = 5,
) -> pd.DataFrame:
    """
    Récupère les données de consommation de gaz depuis l'API ENTSO-G.

    Parameters:
    -----------
    date_start : str, optional
        Date de début au format "YYYY-MM-DD".
        Par défaut: 90 jours avant aujourd'hui.

    date_end : str, optional
        Date de fin au format "YYYY-MM-DD".
        Par défaut: aujourd'hui.

    point_type_id : int, default=18
        ID du type de point (18 = Aggregated Point - Final Consumers).
        Autres valeurs possibles: voir documentation ENTSO-G.

    period_type : str, default="day"
        Type de période ("day", "month", "year").

    timezone : str, default="CET"
        Fuseau horaire (CET, UTC, etc.).

    output_dir : str, default="data/raw"
        Dossier de sortie pour le fichier CSV.

    chunk_days : int, default=180
        Taille des sous périodes en jour.
        A pour but d'éviter les timeout error

    timeout : int, default=180
        Timeout par appel, au lieu de 60s en dur.

    max_retries : int, default=3
        Nombre maximum de tentatives par chunk en cas d'erreur transitoire
        (timeout, 504 Gateway Time-out, etc.).

    retry_wait : int, default=5
        Nombre de secondes d'attente entre deux tentatives.

    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les données de consommation.

    Examples:
    ---------
    >>> # Récupérer les 3 derniers mois
    >>> df = get_consumption_data()

    >>> # Récupérer une période spécifique
    >>> df = get_consumption_data(
    ...     date_start="2024-01-01",
    ...     date_end="2024-12-31"
    ... )
    """


    # 1. Gestion des dates par défaut

    if date_end is None:
        date_end = datetime.now().strftime("%Y-%m-%d")

    if date_start is None:
        # Par défaut: 90 jours avant la date de fin
        date_end_obj = datetime.strptime(date_end, "%Y-%m-%d")
        date_start_obj = date_end_obj - timedelta(days=90)
        date_start = date_start_obj.strftime("%Y-%m-%d")

    logger.info(f"Récupération des données de {date_start} à {date_end}")


    # 2. Construction de l'URL API

    base_url = "https://transparency.entsog.eu/api/v1/operationaldata.csv"
    logger.info(f"URL de base: {base_url}")

    # 3. Découpage de la période en sous-périodes (chunks)
    # Pour éviter les timeouts de l'API ENTSO-G sur les longues périodes,
    # on découpe la requête en plusieurs appels successifs.
    from io import StringIO
    date_start_obj = datetime.strptime(date_start, "%Y-%m-%d")
    date_end_obj = datetime.strptime(date_end, "%Y-%m-%d")
    chunks = []
    chunk_start = date_start_obj
    while chunk_start <= date_end_obj:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), date_end_obj)
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = chunk_end + timedelta(days=1)

    logger.info(f"Période découpée en {len(chunks)} chunk(s) de ~{chunk_days} jours")

    # 4. Boucle d'appels API + chargement de chaque chunk dans un DataFrame
    dfs = []
    for i, (c_start, c_end) in enumerate(chunks, start=1):
        params = {
            "forceDownload": "true",
            "isTransportData": "true",
            "idPointType": point_type_id,
            "delimiter": "comma",
            "indicator": "Physical Flow",
            "from": c_start,
            "to": c_end,
            "periodType": period_type,
            "timezone": timezone,
            "periodize": "0",
            "dataset": "1",
            "limit": "-1",
        }

        logger.info(f"[Chunk {i}/{len(chunks)}] {c_start} → {c_end} | envoi de la requête...")

        # Appel API du chunk courant avec retry sur erreur transitoire (timeout, 504, ...)
        import time
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()  # Lève une exception si erreur HTTP
                logger.info(f"[Chunk {i}/{len(chunks)}] Statut: {response.status_code}")
                break
            except requests.exceptions.Timeout:
                logger.error(f"[Chunk {i}/{len(chunks)}] Tentative {attempt}/{max_retries}: timeout (>{timeout}s).")
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status is not None and 500 <= status < 600:
                    logger.error(f"[Chunk {i}/{len(chunks)}] Tentative {attempt}/{max_retries}: erreur serveur {status}.")
                else:
                    logger.error(f"[Chunk {i}/{len(chunks)}] Erreur HTTP non récupérable: {e}")
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"[Chunk {i}/{len(chunks)}] Erreur réseau: {e}")
                raise

            if attempt < max_retries:
                logger.info(f"[Chunk {i}/{len(chunks)}] Nouvelle tentative dans {retry_wait}s...")
                time.sleep(retry_wait)
            else:
                logger.error(f"[Chunk {i}/{len(chunks)}] Échec après {max_retries} tentatives. Essayez un chunk_days plus petit.")
                raise

        # Parsing CSV du chunk courant
        try:
            df_chunk = pd.read_csv(StringIO(response.text))
            logger.info(f"[Chunk {i}/{len(chunks)}] {len(df_chunk)} lignes chargées")
            dfs.append(df_chunk)
        except pd.errors.ParserError as e:
            logger.error(f"[Chunk {i}/{len(chunks)}] Erreur parsing CSV: {e}")
            logger.error(f"Contenu reçu (premiers 500 caractères): {response.text[:500]}")
            raise

    # 5. Concaténation de tous les chunks en un seul DataFrame
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    logger.info(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    logger.info(f"Colonnes: {list(df.columns)}")


    # 6. Génération du nom de fichier
    # Si output_dir est un chemin relatif, on l'ancre à la racine du projet
    # (parent de gas_intelligence/) pour que la sauvegarde fonctionne quel que
    # soit le répertoire d'exécution (notebook, script, etc.).
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"consumption_{date_start}_to_{date_end}_{period_type}.csv"
    filepath = output_path / filename


    # 7. Sauvegarde en CSV

    try:
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Fichier sauvegardé: {filepath}")
        logger.info(f"Chemin absolu: {filepath.resolve()}")

    except IOError as e:
        logger.error(f"Erreur sauvegarde fichier: {e}")
        raise

    return df
