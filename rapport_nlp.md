# Classification des Urgent Market Messages (REMIT/UMM) : Intérêt et Méthodologie

## 🎯 Intérêt de cette analyse

Le marché européen du gaz est piloté en temps quasi-réel par un flot de **messages d'urgence** publiés par les opérateurs d'infrastructure (TSO comme GRTgaz, GASCADE, Eustream, Snam, GTS, ou terminaux LNG comme Gate, Zeebrugge, Dunkerque). Ces messages (que ce soit **REMIT Inside Information** ou **ENTSOG Urgent Market Messages (UMM)**) annoncent toute information susceptible d'avoir un effet sur le prix et sur les positions du portefeuille : interruption de capacité, maintenance, incident technique, force majeure.

Le problème : ces messages arrivent **en texte libre**, à toute heure, en anglais, et un trader humain ne peut pas tous les lire à temps. Or **tous les messages ne se valent pas**.

*NB : l'utilisation d'un modèle similaire existe déjà en entreprise, envoyant une notification teams à plusieurs personnes à chaque message de REMIT/ENTSOG pouvant avoir un impact sur les positions de l'entreprise. Les critéres de flag sont surement différents, ce projet étant réalisé dans le cadre d'une démarche pédagogique l'objectif n'est pas reproduire le modèle existant mais de voir comment mettre en place un modèle similaire en utilisant des techniques de NLP vues en classe*

### **Pourquoi classifier ?**

#### 1️⃣ **Tous les UMM ne déplacent n'ont pas le même impact sur le prix (TTF)**

| Type de message | Impact typique sur le TTF spot | Anticipation | Réaction du marché |
|---|---|---|---|
| **Maintenance planifiée** (annoncée des semaines ou même des mois à l'avance) | Quasi nul (déjà dans les prix) | ✅ Pricée à la publication initiale | Aucune surprise |
| **Incident soudain** (panne, fuite, explosion, force majeure) | **Fort, immédiat** (+5 à +30 % intraday possible) | ❌ Choc d'information pur | Spike de volatilité, Repricing des forward |

Distinguer automatiquement et de manière fiable les deux types, c'est isoler le **flux d'information à fort contenu de surprise** : celui qui mérite une alerte au desk de trading.

#### 2️⃣ **Anticiper la volatilité réalisée du prix (TTF)**

Empiriquement, les pics de volatilité sur le TTF Day-Ahead suivent souvent une **rafale d'UMM de type incident** (cf. épisodes Freeport LNG 2022, Nord Stream sabotage 2022, fuite Balticconnector 2023). Un classifieur fiable permet :
- de construire un **indicateur quotidien de stress** (count d'incidents publiés dans la fenêtre 24 h),
- de tester sa **corrélation avec la volatilité intraday** du TTF (Brownian bridge, range Parkinson, etc.),
- éventuellement de **prédire** la volatilité de J+1 à partir du flux REMIT de J.

#### 3️⃣ **Filtrer le bruit pour les modèles fondamentaux de prix**

Un modèle fondamental gas (qui croise stocks, températures, flux pipeline, importations LNG) ne peut pas ingérer 200 messages texte par jour. En revanche, il peut consommer une **série temporelle agrégée** :
- `nb_incidents_par_jour`
- `capacite_off_due_aux_incidents_Mcm`
- `nb_pays_impactes`

Le classifieur de cette étude est la **brique en amont** de ces features. Sans étiquetage automatique, impossible de construire ces séries.

#### 4️⃣ **Cas d'usage opérationnels immédiats**

- **Alerting desk de trading** : push notification dès qu'un message classé "incident" est publié sur un actif sensible (Mallnow, IUK, Bacton, etc.).
- **Backtest narratif** : retracer a posteriori les jours de forte volatilité pour vérifier qu'un incident UMM les précède.
- **Compliance & audit** : conserver une trace structurée des informations matérielles diffusées sur le marché.

---

## 📚 Construction d'un faux dataset (100 messages)

Aucun corpus public, prêt à l'emploi, n'agrège les UMM européens avec un label binaire. Nous avons donc construit un **dataset synthétique de 100 messages** courts, en anglais, inspirés du vocabulaire et de la syntaxe réellement observés sur les plateformes REMIT Inside Information, ENTSOG UMM et EEX Transparency. Ce dataset a été obtenu grâce à Gemini 3.1 Pro, à partir d'un prompt contenant des exemples réels et des règles de génération précises.

### **Règles de génération**

| Règle | Justification |
|---|---|
| **50 messages "planifiés" (label 0) + 50 messages "incidents" (label 1)** | Classes parfaitement équilibrées → pas besoin de SMOTE / class_weight, métriques (accuracy, F1) directement lisibles. |
| **Messages courts (1 à 3 phrases, ~20–35 mots)** | Conforme à la réalité : un UMM est un télex, pas un rapport. |
| **Vocabulaire métier authentique** | On utilise les noms réels d'actifs (Mallnow, OPAL, BBL, IUK, Bergermeer, Velke Kapusany, Gate Terminal, etc.), les unités du marché (Mcm/d, GWh/h), les codes d'équipement (PSV-203, K-302, T-1), les acronymes (ESD, ETR, TSO, force majeure). |
| **Diversité des types d'événements** | Côté planifié : pigging, hydrotest, overhaul turbine, SCADA upgrade, recalibration métering, audit TÜV/SF6, drill émergency. Côté incident : trip compresseur, fuite, force majeure, surchauffe, foudre, third-party damage, ESD, perte SCADA, ice formation, etc. |
| **Distinction *subtile*, pas keyword-based** | C'est le cœur du parti pris pédagogique. Voir détail ci-dessous. |

### **Le piège évité : la distinction par mot-clé**

Une approche naïve dirait : "si le message contient le mot **planned** ou **scheduled** → label 0, sinon label 1". C'est **trop facile** et le modèle ne saurait rien généraliser à de vrais messages d'EEX ou de GRTgaz qui utilisent un vocabulaire plus varié.

Pour forcer le modèle à apprendre **la sémantique** et pas un dictionnaire d'une dizaine de mots, le dataset contient volontairement :

#### **a) Des messages "planifiés" sans mot-clé évident**
```
"Annual pigging operation on the OPAL pipeline. Booked capacity reduced by 25% between 12 and 18 September."
"Pre-winter check of storage facility Bergermeer. Withdrawal capacity temporarily capped at 850 GWh/d from 09/10 to 11/10."
"Routine integrity scan along the Rehden-Edmonton segment. No capacity restrictions notified."
"Annual emergency-response training exercise at IP Vyhne. No real interruption, communications drill only."
```

#### **b) Des messages "incidents" sans mot-clé évident**
```
"Trip of compressor K-4 at IP Velke Kapusany. Interruption ongoing, ETR pending."
"Curtailment at Bocholtz IP after pressure anomaly, cause being investigated."
"Interruption of flows at Greifswald after detection of process upset, corrective maintenance underway."
"Loss of injection capability at storage facility Reden after valve malfunction."
```

#### **c) Du chevauchement lexical délibéré**

Les deux classes partagent des mots ambigus : `outage`, `reduction`, `capacity`, `interruption`, `shutdown`, `maintenance`. Un mot **n'est pas un label** : c'est la combinaison qui fait l'étiquette.

### **Comment un être humain fait pour détecter un message nécessitant une action urgente ?**


**Lexique** : Un analyste classe un message UMM en repérant instantanément des mots-clés d'urgence (panne, incident, imprévu) ou de gestion de projet (planifié, annuel, maintenance).

**Contexte** : L'humain évalue ensuite l'impact immédiat en mesurant si la baisse de capacité annoncée est subie (crise) ou anticipée (routine).

**L'objectif du modèle** : Notre objectif est de vérifier si un modèle de Machine Learning (TF-IDF + Régression Logistique) peut cartographier automatiquement ce lexique et reproduire cette logique de classification.

---

## 🔬 Méthodologie : pipeline NLP supervisé classique

Le pipeline est volontairement **simple** (pas de deep learning, pas de embeddings pré-entraînés) car :
- 100 messages = trop peu pour fine-tuner un transformer sans overfit massif,
- on cherche à **comprendre** ce que le modèle apprend, ce qui est trivial avec un linéaire et impossible avec un BERT.

```
Raw message (string)
        │
        ▼
Text cleaning

        │
        ▼
TfidfVectorizer (sans stop-words, ngram_range=(1,2))
- vocabulaire d'environ 300-600 tokens unigrammes + bigrammes
- pondération TF * IDF
        │
        ▼
Classifieur linéaire (LogisticRegression)
- probabilité calibrée (sigmoid)
- coefficients interprétables = importance par mot
        │
        ▼
Évaluation hold-out (train 80% / test 20%, stratifié)
        │
        ▼
Feature importance (top-15 par classe)
        │
        ▼
Test sur de vrais UMM EEX / GRTgaz
```

---

## 📊 Pipeline méthodologique (6 étapes)

### **Étape 1 — Preprocessing du texte** 🔧

**Objectif** : ramener chaque message à une forme **canonique** pour que le vectoriseur ne distingue pas "Mallnow" et "mallnow", ou ne crée pas un token par numéro de borne kilométrique.

| Opération | Pourquoi |
|---|---|
| `lower()` | "Dunkerque" et "dunkerque" sont le même token. |
| Suppression des chiffres et ponctuation | Les chiffres (heures, dates, valeurs en Mcm/d) sont du bruit : ils ne discriminent pas planifié vs incident. |
| Pas de stemming / lemmatisation | Vocabulaire technique court : "failure" et "failed" cohabitent, l'utilisation de bigrammes TF-IDF compense. |
| **Pas** de stop-word removal | Sur des messages aussi courts les stopwords comme "of the" ou "due to" peuvent porter du signal (ex : "due to a sudden technical failure" est un marqueur fort). |

**Sortie** : 100 chaînes nettoyées, prêtes pour le TF-IDF.

---

### **Étape 2 — Extraction de features : TF-IDF** 📊

Après une longue réflexion et quelques essais, nous avons décidé de ne présenter ici que l'utilsation de TF-IDF.

**Pourquoi TF-IDF et pas un simple Bag-of-Words ?**

| Méthode | Force | Limite sur notre corpus |
|---|---|---|
| Bag-of-Words (CountVectorizer) | Simple, direct | Surpondère les mots fréquents non informatifs (`the`, `at`) |
| **TF-IDF** | Pénalise les mots qui apparaissent partout, valorise les mots discriminants | ✅ Adapté |
| Word embeddings (Word2Vec, GloVe) | Capture la sémantique | Inutile sur 100 messages : pas de signal à apprendre |
| Transformers | État de l'art (surement utilisé en entreprise) | Sur-dimensionné, opaque, et overfit garanti sur 100 lignes |

**Sortie** : matrice creuse (sparse matrix) `100 × V` où `V ≈ 250-300` tokens (correspond au nombre de /bigrames différents dans le corpus).

---

### **Étape 3 — Modélisation : LogisticRegression** 🤖

**On retient `LogisticRegression`** parce que :
1. La visualisation de feature importance (objectif clé) est **directement lisible** : `coef_[0]` est l'importance par mot, signe inclus.
2. Le corpus est petit, le coût de calcul est faible.
3. Régularisation L2 (par défaut) → robuste sur vocabulaire à `min_df=2`.

Configuration : `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`.

---

### **Étape 4 — Évaluation** ✅

**Protocole** :
- Split stratifié train/test = 80/20 → 80 messages d'entraînement, 20 messages de test (10 planifiés, 10 incidents).
- Métriques rapportées : **accuracy**, **precision/recall/F1 par classe**, **matrice de confusion**.
- Pas de cross-validation k-fold dans le notebook principal (le corpus est trop petit pour qu'une 5-fold soit stable), mais le code reste reproductible (`random_state=42`).

**Analyse des résultats du modèle** :
- Accuracy **80 à 90 %** sur ce dataset volontairement piégeux (en fonction du seed, du split et de `min_df`).
Sur le `random_state=42` retenu, on trouve une accuracy de **80 %** (4 erreurs sur 20).
Après lecture à la main des erreurs résiduelles, on se rend compte qu'il s'agit **des messages-pièges** mentionnés dans la section "Construction du dataset", et leur examen donne une lecture de ce que le modèle ne sait pas faire.
- Cas typiques d'erreur :
  - *"Trip of regasification pump P at Dunkerque LNG. Send-out reduced, no ETR yet."* → mal classé en planifié car le token `no` est l'un des plus forts marqueurs négatifs appris (il apparaît dans "**no** impact", "**no** interruption", "**no** transport restriction" côté planifié) et masque le signal incident porté par `trip` / `etr`.
  - *"Pipeline rupture detected between KP X and KP Y. Section isolated, repair team mobilised."* → ambigu car `repair team mobilised` a une tonalité calme et procédurale.

**Comprendre où le modèle échoue** est aussi important que la performance brute. Ces erreurs nous indiquent que le modèle est encore **lexical** et **mono-token-dominant**. Pour aller plus loin, soit augmenter le dataset (cf. roadmap V2), soit ajouter un encodeur contextuel (DistilBERT) qui saurait pondérer `no` différemment dans "no ETR yet" vs "no impact expected".

Pour comprendre comment le modèle décide passons à une analyse détaillé des features ayant un grand impact sur la classification.

---

### **Étape 5 — Feature Importance** 💡

C'est **la visualisation centrale** du notebook. Pour un `LogisticRegression` binaire avec classe positive = `1` (incident) :

- `coef_[i] > 0` → le token `i` pousse la prédiction vers **incident**.
- `coef_[i] < 0` → le token `i` pousse la prédiction vers **planifié**.

On affiche un **barplot horizontal** avec :
- Les **15 mots/bigrammes les plus positifs** (rouge) → indicateurs d'incident.
- Les **15 mots/bigrammes les plus négatifs** (vert) → indicateurs de planification.

**Ce qu'on peut voir** :

| Probable côté incident (rouge) | Probable côté planifié (vert) |
|---|---|
| trip, failure, fault, leak, fire, alarm, anomaly, force, force majeure, etr, sudden, cause, malfunction, lightning, rupture | annual, scheduled, planned, routine, maintenance, inspection, overhaul, periodic, audit, pre-winter, capacity reduced, from to, hours starting |

**Une sémantique métier parfaitement apprise** : Le modèle démontre une excellente capacité à isoler le champ lexical de l'imprévu (unplanned, sudden, unexpected, failure) pour la classe des incidents, prouvant qu'il ne se base pas sur du bruit mais sur de vrais signaux d'alerte opérationnelle.

**Une distinction claire de la routine** : Du côté des événements planifiés, le modèle s'appuie logiquement sur des concepts d'organisation temporelle récurrents (annual, scheduled, planned, routine), ce qui correspond exactement à la structure attendue des maintenances programmées sur le marché gazier.

**Potentielle piste d'amélioration** : La présence de petits mots de liaison (comme to, from, at) confirme que le nettoyage du dictionnaire (stop-words) peut être nécessaire pour forcer la régression logistique à devenir un outil d'analyse purement axé sur le sens technique plutôt que sur le style syntaxique.

---

### **Étape 6 — Test en conditions réelles** 🌍

Le dataset d'entraînement a beau être réaliste, il reste **synthétique**. Le risque classique est que le modèle ait appris des **tics de style** propres à Gemini (ponctuation, longueur, formulation), et non la sémantique d'un UMM.

On a donc créer la fonction `classify_messages(texts)`:

```python
def classify_messages(texts: list[str]) -> pd.DataFrame:
    """Renvoie pour chaque texte la prédiction et la probabilité d'incident."""
```

L'utilisation prévue est de **copier-coller en direct** quelques vrais messages depuis :
- [EEX Transparency](https://www.eex-transparency.com/) — section Gas → REMIT / UMM,
- [GRTgaz Operational Information](https://www.grtgaz.com/) — section actualité opérationnelle,
- [ENTSOG Transparency Platform](https://transparency.entsog.eu/) — UMM,
- [Gassco UMM](https://umm.gassco.no/) — incidents Norvège,

Il se trouve que la classification est cohérente avec mon ressenti humain → la prochaine étape serait alors d'élargir le corpus avec **de vrais UMM annotés à la main**.

---

## 🧭 Limites et axes d'amélioration

### **Ce qui est solide**
- ✅ Pipeline reproductible, interprétable, justifié à chaque étape.
- ✅ Dataset équilibré, vocabulaire métier authentique.
- ✅ Feature importance permettant un dialogue concret avec un humain.

Notre démarche présente certaines lacunes qu'il faudrait explorer. C'est pourquoi nous avons rédiger une Roadmap pour poursuivre le développement de cette approche.
### **Roadmap V2 proposée**
1. **Scraping** des UMM EEX/ENTSOG sur 12 mois → ~10 000 messages bruts.
2. **Annotation** manuelle sur un sous-échantillon de 1 000 messages (multiclasse).
3. **Réentraînement** avec un modèle moyen-corpus : `LogisticRegression + char n-grams` ou `DistilBERT` fine-tuné.
4. **Productisation** : traitement automatique des messages reçus sur EEX/ REMIT.. et envoie de notifications Teams / Bloomberg push dès qu'un message avec proba_incident > 0.75 est reçu.

---
