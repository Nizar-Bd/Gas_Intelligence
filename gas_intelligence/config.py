"""Constantes et paramètres centralisés du pipeline de clustering."""

# ---------------------------------------------------------------------------
# Seuils empiriques pour la classification métier
# ---------------------------------------------------------------------------
CONSO_THRESHOLD_SMALL = 50      # GWh/j  : limite résidentiel petit / moyen
CONSO_THRESHOLD_LARGE = 200     # GWh/j  : limite moyen / gros industriel
VOLATILITY_LOW = 2              # σ/μ    : limite stable / volatile
VOLATILITY_HIGH = 8             # σ/μ    : limite volatile / très volatile
RATIO_HE_LOW = 1.5              # peu saisonnier (< 1.5)
RATIO_HE_HIGH = 2.5             # très saisonnier (> 2.5)
RATIO_WE_WD_LOW = 0.8           # effet weekend marqué (< 0.8)

# ---------------------------------------------------------------------------
# Seuils utilisés pour identifier les outliers à retirer avant clustering
# ---------------------------------------------------------------------------
OUTLIER_CONSO_MEAN = 600        # GWh/j
OUTLIER_VOLATILITY = 15
OUTLIER_RATIO_HE = 0.1          # ratio_hiver_ete < 0.1

# pointKeys identifiés manuellement comme outliers dans le notebook 1
DEFAULT_OUTLIER_KEYS = [
    "FNC-00006",  # Thermal Plants (IT)
    "FNC-00007",  # Gas mixing facilities (PL)
    "FNC-00043",  # Aggregated Distribution + Final Customers Exits (ES)
    "FNC-00206",  # Final Consumers DE GTG Nord H Gas
]

# ---------------------------------------------------------------------------
# Paramètres du clustering
# ---------------------------------------------------------------------------
K_OPTIMAL = 3
RANDOM_STATE = 42
KMEANS_N_INIT = 10

DBSCAN_EPS = 1.2
DBSCAN_MIN_SAMPLES = 3

LINKAGE_METHOD = "ward"
DENDROGRAM_CUT_DISTANCE = 7

# Features utilisées pour le clustering (ordre stable)
CLUSTERING_FEATURES = [
    "conso_mean",
    "volatilite",
    "ratio_hiver_ete",
    "ratio_we_wd",
]

# ---------------------------------------------------------------------------
# Nomenclature métier des clusters
# ---------------------------------------------------------------------------
CLUSTER_NAMES = {
    0: "Saisonnier - Chauffage dominant",
    1: "Atypiques - Petits L-Gas / Tertiaire",
    2: "Industriel / Mixte continu",
}

COLOR_PALETTE = {
    0: "#4C72B0",   # bleu         - Saisonnier (chauffage)
    1: "#C44E52",   # rouge        - Atypiques L-Gas / Tertiaire
    2: "#55A868",   # vert         - Industriel / Mixte continu
    -1: "#808080",  # gris         - bruit DBSCAN
}

# ---------------------------------------------------------------------------
# Définition des mois (saisonnalité)
# ---------------------------------------------------------------------------
WINTER_MONTHS = [1, 2, 3, 10, 11, 12]
SUMMER_MONTHS = [4, 4, 6, 7, 8, 9]
WEEKEND_DAYS = [5, 6]  # samedi, dimanche (lundi=0)

# Valeur d'encodage utilisée pour les ratios infinis (été = 0)
INFINITE_RATIO_FILL = 3

# ---------------------------------------------------------------------------
# Paramètres NLP — classifieur REMIT/UMM (notebook 2)
# ---------------------------------------------------------------------------
NLP_RANDOM_STATE = 42                # seed pour split + LogisticRegression
NLP_TEST_SIZE = 0.20                 # part du hold-out (stratifié)

TFIDF_NGRAM_RANGE = (1, 2)           # unigrammes + bigrammes (capter "force majeure", "as planned")
TFIDF_MIN_DF = 2                     # ignorer les tokens qui n'apparaissent que dans un seul message
TFIDF_LOWERCASE = True               # belt-and-braces (le preprocessing met déjà en minuscules)
TFIDF_STOP_WORDS = None              # consigne projet : on garde les stop-words

LOGREG_C = 1.0                       # régularisation L2 par défaut
LOGREG_MAX_ITER = 1000               # marge confortable pour la convergence

# Chemin par défaut vers le faux dataset REMIT (résolu depuis la racine du projet)
DEFAULT_REMIT_CSV = "data/raw/mock_remit_messages.csv"

# Étiquettes métier pour les deux classes (label binaire 0 = planifié, 1 = incident)
REMIT_LABEL_NAMES = {
    0: "Planifié",
    1: "Incident",
}

# Seuil de probabilité pour qualifier la confiance d'une prédiction "haute" vs "moyenne"
NLP_HIGH_CONFIDENCE_THRESHOLD = 0.85

