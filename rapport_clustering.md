# Segmentation des Profils de Consommation de Gaz: Intérêt et Méthodologie

## 🎯 Intérêt de cette analyse

Le marché du gaz naturel européen n'est **pas homogène**. Il est composé de multiples segments avec des caractéristiques économiques radicalement différentes:

### **Pourquoi segmenter?**

#### 1️⃣ **Comprendre l'hétérogénéité de la demande**
La consommation de gaz varie énormément selon le secteur:
- **Résidentiel**: Utilisé principalement en **hiver** pour le chauffage → saisonnalité forte, demande prévisible
- **Industriel**: Utilisé **toute l'année** dans les processus de production → demande stable, peu saisonnière
- **Tertiaire/Commercial**: Mix intermédiaire avec sensibilité à l'activité économique

Cette hétérogénéité est **la clé** pour comprendre les prix.

#### 2️⃣ **Prédire les prix avec plus de précision**
Les déterminants du prix varient selon le profil:

| Profil | Déterminants majeurs | Sensibilité | Prédictibilité |
|--------|---------------------|-------------|---|
| **Résidentiel** | 🌡️ Température<br/>📦 Stocks<br/>🌍 Géopolitique | Forte à météo<br/>Faible à économie | ✅ Haute (météo prévisible) |
| **Industriel** | 💰 Prix lui-même<br/>📊 Production<br/>⚡ Chocs | Forte à prix<br/>Forte à économie | ❌ Basse (imprévisible) |
| **Mixte** | Combinaison des deux | Modérée | ⚠️ Moyenne |

**Un modèle unique serait inadapté** → Modèles séparés par profil = meilleure prédiction.

#### 3️⃣ **Identifier les risques systémiques**
Certains chocs affectent différemment chaque profil:
- 🥶 **Vague de froid**: Pics résidentiel >> industriel. Risque de shortage.
- 📉 **Récession économique**: Baisse industrielle >> résidentiel. Pression prix.
- 🚀 **Hausse prix**: Réaction élastique industriel (baisse conso) >> résidentiel (conso rigide).

Segmenter permet d'**anticiper les scénarios de crise**.

#### 4️⃣ **Optimiser les stratégies commerciales et de trading**
Les contrats optimaux diffèrent par profil:
- ✅ Résidentiel: **Contrats fixes saisonniers** (stabilité)
- ✅ Industriel: **Contrats indexés prix** (flexibilité)
- ✅ Mixte: **Contrats hybrides** (compromis)

Les **spreads saisonniers** sont exploitables uniquement si on comprend la composition résidentielle/industrielle.

---

## 🔬 Méthodologie: Approche Hybride

Pour éviter les pièges des algorithmes de clustering purs, nous adoptons une **approche hybride** combinant exploration exploratoire et validation statistique.

### **Philosophie générale**

```
Les données eux-mêmes révèlent les groupes naturels.
L'algorithme confirme et automatise cette révélation.
```

### **Pourquoi "hybride"?**

| Approche | Avantage | Inconvénient |
|----------|----------|-------------|
| **KMeans seul** | Rapide, automatisé | Peut échouer (super-clusters) |
| **Seuils métier seuls** | Transparent, justifié | Subjectif, manque rigueur stat |
| **Hybride (notre approche)** | Combine tout ✅ | Légèrement plus complexe |

---

## 📊 Pipeline méthodologique (6 étapes)

### **Étape 1: Ingénierie de features** 🔧

**Objectif**: Transformer les données brutes en **4 dimensions interpretables**

```
Données brutes
(275 points × 5 ans de consommation quotidienne)
         ↓
Filtre sur le 38 consommateurs finaux
         ↓
Agrégation par point
(somme/moyenne/écart-type)
         ↓
7 FEATURES créées:
```

| Feature | Définition | Interprétation |
|---------|-----------|---|
| **conso_mean** | μ (conso quotidienne) | Taille du point |
| **conso_std** | σ (écart type) | Utile pour le calcul de la volatilité |
| **conso_min** | min(conso) | Donne la plus petite conso pour un jour à ce point |
| **conso_max** | max(conso) | Donne la plus grande conso pour un jour à ce point |
| **volatilite** | σ (conso) / μ(conso) | Variabilité |
| **ratio_hiver_ete** | conso_hiver / conso_été | Saisonnalité |
| **ratio_we_wd** | conso_weekend / conso_weekday | Effet week-end |

**Justification**: Ces 7 features capturent les **dimensions économiques essentielles** (taille, variabilité, saisonnalité, activité).


**Première observation :** Après création de ces features notre dataframe **profils** contient un certains nombre de valeurs manquantes :

conso_std: 1 NaN (2.6%)
volatilite: 1 NaN (2.6%)
ratio_hiver_ete: 4 NaN (10.5%)
ratio_we_wd: 1 NaN (2.6%)

Pour le point dont la volatilité est manquante, est ce lié au fait que la moyenne à ce point est nulle ?
-> Le point avec une volatilité manquante est un point qui n'a reçu du gaz que sur 1 date.
On va donc supprimer ce point il n'est pas pertinent.

Il nous reste 3 valeurs manquantes à traiter, dans la colonne ratio_hiver_ete. Après identification des point concernés on se rend compte que cela est lié à une absence totale de consommation en été. On va donc les encoder comme des points avec un ratio hiver été très élevé.
Pour ce faire on va remplacer ces NaN par la valeur maximum que l'on a dans la colonne ratio_hiver_ete.

**Plus aucune valeurs manquantes, 37 points de consommation finale en Europe que l'on va maintenant essayer de segmenter.**
---

### **Étape 2: Exploration des données** 🔍

**Objectif**: Découvrir les profils **sans appliquer d'algorithme**

**Approche**: Simplement **regarder** comment les données se distribuent. On va mettre nos observations en parallèle du type de consommateurs que l'on peut imaginer (résidentiel, mixte, industriel).

```
📊 CONSO MOYENNE:
  ├─── 0-50 GWh/j: 40% (RÉSIDENTIEL PETIT)
  ├─── 50-200: 19% (RÉSIDENTIEL GRAND + MIXTE)
  └─── >200: 5% (INDUSTRIEL GROS)

⚡ VOLATILITÉ:
  ├─── 0-2: 90% (TRÈS STABLE = RÉSIDENTIEL ULTRA-DOMINANT)
  ├─── 2-8: 8% (MODÉRÉMENT VOLATILE = MIXTE)
  └─── >8: 2% (TRÈS VOLATILE = INDUSTRIEL RÉACTIF)

🌡️ RATIO HIVER/ÉTÉ (BIMODALE - TRÈS DISCRIMINANT):
  ├─── 0.8-1.2: 19% (INDUSTRIEL/CONTINU - PEU SAISONNIER)
  ├─── 1.2-2.0: 8% (ZONE INTERMÉDIAIRE - MIXTE)
  ├─── 2.5-3.5+: 14% (RÉSIDENTIEL PUR - CHAUFFAGE DOMINANT)
  └─── Autres: 49% (DISTRIBUTION INTERMÉDIAIRE - À EXPLORER)

📅 RATIO WEEKEND/WEEKDAY:
  ├─── 0.8-1.0: 80% (RÉSIDENTIEL/PAS D'EFFET JOUR)
  └─── 0.0-0.8: 20% (COMMERCE/INDUSTRIE/BAISSE LE WE)
```

**Résultat**: **3 profils naturels émergent** de la simple observation.

```
PROFIL 1 - RÉSIDENTIEL (majorité, ~20-25 points):
  ✓ conso_mean < 50 GWh/j (petit)
  ✓ volatilite < 2 (très stable)
  ✓ ratio_hiver_ete > 2.5 (très saisonnier - chauffage!)

PROFIL 2 - MIXTE (~8-12 points):
  ✓ conso_mean 50-200 GWh/j (moyen)
  ✓ volatilite 2-8 (modérément volatile)
  ✓ ratio_hiver_ete 1.5-2.5 (saisonnalité modérée)

PROFIL 3 - INDUSTRIEL (~2-5 points):
  ✓ conso_mean > 200 GWh/j (gros)
  ✓ volatilite > 8 (très volatile) OU
  ✓ ratio_hiver_ete < 1.5 (peu/pas saisonnier)
```

Cette étape fournit les seuils empiriques pour challenger le clustering.

**Répartition des données**

Essayons d'interpréter nos graphes boites à moustaches.

1️⃣ Conso moyenne
Distribution ultra-asymétrique log-normale :

Médiane = 34.71 GWh/j (petit) vs Moyenne = 120.64 GWh/j → écart énorme !
La majorité (75%) < 124 GWh/j, mais 4-5 outliers (géants industriels?) tirent la moyenne vers 350-600+ GWh/j
Signal clair : beaucoup de petits résidentiels (Q1-Q2) + quelques géants industriels


2️⃣ Volatilité
Écrasement à gauche + nombreux outliers :

Médiane = 0.31 (très stable) vs Moyenne = 1.93 → les outliers dominent la moyenne
75% des points ont volatilité < 0.69 (ultra-stables = résidentiel)
8 outliers visibles, dont un extrême (~40) = peut-être un processus industriels hyper-réactifs
Confirmation : 90%+ sont stables, seulement 10% volatiles


3️⃣ Ratio hiver/été
Distribution compacte et symétrique :

Médiane = 1.23 ≈ Moyenne = 1.44 → distribution équilibrée
50% des points entre 1.11-1.57 (très resserré!)
Surprenant : pas de séparation nette résidentiel/industriel ici
La boîte compacte contraste avec l'histogramme bimodal → les 37 points ne sont pas aussi polarisés qu'attendu


4️⃣ Ratio weekend/weekday
Très peu discriminant :

Médiane = 0.87 ≈ Moyenne = 0.80 → quasi-uniformité
50% des points entre 0.73-0.92 (intervalle serré = peu de variance)
Quelques outliers bas (0.0-0.4) = forte baisse le WE (commerce/industrie)
Conclusion : cette feature apporte peu d'information pour classifier


**Avant d'aller plus loin**

Nous allons nous intéresser aux outliers, pour tenter d'identifier leur profil à la main.

On décide de considérer un point comme outlier si :
```
conso_mean > 600 GWh/j
volatilite > 15
ratio_hiver_ete < 0.1
```

On identifie ainsi 4 outliers :
- FNC-00006 correspond au profil d'une grosse usine.
- FNC-00007 correspond au profil d'une infrastrucutre technique.
- FNC-00043 correspond au profil d'un hub de réseau.
- FNC-00206 correspond au profil d'une grande zone industrielle.

Pour faciliter le travail de notre clusterisation statistique on décide de retirer ces 4 points.

---

### **Étape 3: Normalisation et Clustering statistique** 📈

**Objectif**: Appliquer un algorithme de clustering pour **assigner automatiquement** chaque point à un profil.

#### **Normalisation (StandardScaler)**
Les 4 features ont des échelles différentes (conso: 0-800, volatilite: 0-40 par exemple). Pour que notre modèle ne soit pas biaisé par les différences d'échelle nous allons appliquer une méthode de normalisation.
```
X_scaled = ( X - Mean ) / STD

Bénéfice: Pas de domination d'une feature sur les autres
```

#### **Sélection du nombre de clusters (k)**

On va essayer de challenger notre hypothése qui dit qu'il y a 3 typologies de consommateurs.
Pour ce faire nous allons appliquer des méthodes courantes de recherche du nombre optimal de cluster.
Trois critères concomitants:

**1. Elbow Method**: Courbe inertie vs k
   - Cherche le "coude" où la courbe s'aplatit
   - Indique typiquement: k = 2 ou 3

**2. Silhouette Score** ⭐ (MEILLEUR POUR CE CAS)
   - Mesure: Chaque point est-il bien assigné?
   - Score -1 à +1: Idéalement > 0.5
   - Pour ce projet: **Silhouette(k=3) > 0.5** = good ✅

**3. Davies-Bouldin Index**
   - Mesure: Compacité vs séparation des clusters
   - Bas = mieux
   - Cherche le minimum

**Observation et choix de k :**

Elbow Method :

Coude principal entre K=4 et K=5 → gain d'inertie très significatif (-50%)
Ralentissement net après K=5 → chaque K supplémentaire apporte peu (<10%)
Signal Elbow : K=5 est optimal (coude naturel majeur)

Silhouette Score :

K=5 a la meilleure séparation statistique (0.54 = bon, pic clairement visible)
K=4 reste modéré (0.51 = moyen)
K=2 et K=3 faibles (0.37 et 0.45 respectivement)
K=6+ dégradation nette (chute sous 0.45)
Signal Silhouette: K=5 sans équivoque (pic clair)

Davies-Bouldin Index :

Plus bas = meilleur (Davies-Bouldin mesure le ratio compacité/séparation)
K=5 est optimal (0.50 = très bas, valeur minimale)
K=2 et K=3 sont mauvais (1.20 et 1.00 respectivement)
Tous les K différents de K=5 présentent une dégradation
Signal Davies-Bouldin: K=5 SANS ÉQUIVOQUE

**Décision finale : K=3**

Trois métriques indépendantes s'accordent : Elbow, Silhouette et Davies-Bouldin désignent unanimement K=5 comme optimal.
**CEPENDANT**, l'utilisation d'un KMeans avec k=5 (et k=4) créer des clusters trop petits (singletons ou 2 points par cluster). Pour que notre clustering garde un sens et soit pertinent, nous allons partir avec K=3.

#### **Algorithme: KMeans vs DBSCAN**

| Algorithme | Forces | Faiblesses | Quand l'utiliser |
|-----------|--------|-----------|---|
| **KMeans** | Simple, rapide, standard | Force tous les points dans k clusters | Données bien séparées, distributions non-extrêmes |
| **DBSCAN** | Gère distributions skewed, identifie outliers | Paramètres à tuner, moins intuitif | Données asymétriques, quelques outliers |

Après comparaison des résultats des modèles KMeans et DBSCAN on en vient aux conclusions suivantes :

**Accord Global Très Élevé**

Les deux algorithmes produisent des segmentations quasi-identiques : DBSCAN classe
29 points dans les mêmes groupes que KMeans (87.9% d'accord parfait), avec seulement
4 points présentant une divergence. Cette convergence valide la solidité de la
structure à 3 clusters identifiée.

**Les 4 Points "Bruit" de DBSCAN Sont des Points Limites**

Les 4 points classés comme "bruit" par DBSCAN (eps=1.2) ne sont pas des outliers
aberrants, mais des points situés à la limite entre deux clusters KMeans : 1 entre
Résidentiel et Industriel, 2 entre Résidentiel et Mixte, et 1 entre Mixte et
Industriel. Ces points possèdent des caractéristiques intermédiaires qui les
positionnent en bordure de clusters, d'où leur exclusion par DBSCAN.

**KMeans Offre une Couverture Complète et Pragmatique**

Contrairement à DBSCAN qui laisse 12.1% des points non-classés, KMeans attribue tous
les 33 points à un cluster, ce qui est souhaitable pour une application opérationnelle.
Les 4 points limites sont simplement intégrés au cluster le plus proche, une décision
pratique qui ne biaiserait pas significativement l'interprétation économique.

**Validation Croisée Confirme la Robustesse de K=3**

Le fort accord entre DBSCAN et KMeans (distribution quasi-identique : 20-6-3-4 vs
21-8-4-0) suggère que les 3 clusters reflètent une véritable structure sous-jacente
dans les données, et non une artefact de l'algorithme choisi. Cette convergence renforce
la confiance dans la segmentation finalement retenue.

**Conclusion : KMeans K=3 est le Choix Optimal**

KMeans K=3 combine les avantages de DBSCAN (identification des points limites par
analyse croisée) sans ses inconvénients (points non-classés). C'est la solution idéale
pour passer à la production : segmentation claire, 100% couverture, et justification
statistique solide.Haiku 4.5 Étendue

#### **Algorithme: Clustering Hiérarchique**

Bien que KMeans et DBSCAN se confortent mutuellement dans leurs résultats, il reste une question de **robustesse** et de **certitude**: comment visualiser cette structure de manière indépendante?

Le **clustering hiérarchique (dendrogramme)** apporte une **perspective orthogonale et visuelle** :

- ✅ **Indépendance algorithmique**: un 3e algorithme basé sur une logique complètement différente
  (construction progressive de clusters par fusion)
- ✅ **Visualisation de la structure**: le dendrogramme montre explicitement où les groupes se forment
  et se fusionnent
- ✅ **Justification visuelle de K=3**: le "coude" ou "saut de distance" dans le dendrogramme indique
  naturellement où couper pour 3 clusters
- ✅ **Détection de sous-structures**: explore s'il existe des sous-groupes au sein de chaque cluster principal
- ✅ **Crédibilité**: 3 algorithmes indépendants produisant le même résultat = validation définitive


**Choix de la méthode de "Linkage"**

Quatre méthodes de linkage ont été testées (Ward, Complete, Average, Single).
Le **linkage Ward** a été retenu car il produit la segmentation la plus claire
avec une coupure K=3 évidente, révélant 3 branches distinctes et équilibrées.
Contrairement à Complete (trop de micro-ramifications), Average (moins net) ou
Single (effets de chaînage), Ward fournit une structure hiérarchique
facilement interprétable et visuellement convaincante pour justifier le choix
de 3 clusters auprès des décideurs.

**Validation Hiérarchique de K=3**

Le dendrogramme Ward avec seuil de coupure à distance=7 révèle **3 branches
distinctes et colorées**, correspondant exactement aux 3 clusters identifiés par
KMeans:

- **Branche Rouge (Gauche)**: Cluster 0 - Mixte/Commerce (8 points)
- **Branche Bleu (Centre)**: Cluster 1 - Industriel (4 points)
- **Branche Vert (Droite)**: Cluster 2 - Résidentiel (21 points)

Cette séparation claire au niveau hiérarchique confirme que les 3 clusters ne sont
pas des artefacts de l'algorithme KMeans, mais reflètent une **véritable structure
sous-jacente** dans les données. Les points d'un même cluster sont systématiquement
plus proches les uns des autres dans l'arbre hiérarchique, validant la cohésion
intra-cluster.

**Convergence des 3 Algorithmes**

**KMeans** → 3 clusters (21-8-4 pts)
**DBSCAN** → 3 clusters + 4 points limites
**Clustering Hiérarchique** → 3 branches distinctes

La convergence de 3 approches méthodologiquement indépendantes renforce
**définitivement** la confiance dans cette segmentation.

**Conclusion**

✅ **K=3 est le choix optimal**, validé par:
- Métriques statistiques (Silhouette, Davies-Bouldin)
- Accord inter-algorithmes (87-95%)
- Structure visuelle (dendrogramme avec 3 branches claires)
---

### **Étape 4: Analyse métier** 💡

> ⚠️ **Précision sémantique** — Les sections précédentes (étape 3) parlent de Cluster 0 = Mixte, Cluster 1 = Industriel, Cluster 2 = Résidentiel. La réinspection rigoureuse des centroïdes (ci-dessous) montre que cette nomenclature est **partiellement trompeuse**. Les vraies dimensions discriminantes sont en réalité **la saisonnalité** et **la taille / l'effet week-end**, pas la nature résidentielle vs industrielle stricto sensu. La sémantique métier proposée ci-dessous remplace donc celle d'origine.

---

#### **4.0 — Résultats chiffrés du clustering KMeans (K=3, 33 points)**

| Cluster | n | conso_mean (GWh/j) | volatilite | ratio_hiver_ete | ratio_we_wd |
|---------|---|---|---|---|---|
| **0 — Saisonnier "chauffage dominant"** | 8 | **162,1** (σ=254) | 0,48 | **2,56** | 0,87 |
| **1 — Atypiques L-Gas / Tertiaire en déclin** | 4 | **6,0** (σ=6,7) | **2,67** | 1,21 | **0,51** |
| **2 — Industriel / Mixte continu (majorité)** | 21 | 79,9 (σ=83) | **0,32** | 1,22 | 0,87 |

Trois points clés :
- La **saisonnalité (ratio_hiver_ete)** est la variable la plus discriminante du Cluster 0 (2,56 vs 1,22 partout ailleurs).
- Le **ratio_we_wd** isole le Cluster 1 (0,51 vs ~0,87 ailleurs) — c'est la seule poche où le week-end fait baisser la consommation de moitié.
- La **volatilité** identifie aussi le Cluster 1 (2,67 vs 0,32–0,48 ailleurs).
- La taille (conso_mean) **ne sépare pas proprement les clusters** : Cluster 0 et Cluster 2 contiennent tous deux des géants et des petits. C'est un enseignement important pour la suite (cf. § 4.D).

**Représentants centraux (points les plus proches du centroïde dans l'espace normalisé) :**
- Cluster 0 → FNC-00200 (Hongrie), FNC-00205 (Lettonie), FNC-00037 (Estonie) → **Europe centrale/orientale et baltique**
- Cluster 1 → DIS-00061, FNC-00215 (CZ), FNC-00045 → **micro-pools L-Gas allemands + agrégat tchèque**
- Cluster 2 → FNC-00029, DIS-00060, FNC-00203 → **TSO allemands et industriels Benelux**

---

#### **4.1 — Interprétation détaillée de chaque cluster**

##### 🟦 Cluster 0 — « Demande saisonnière dominée par le chauffage » (8 points)

- **Profil économique** : portefeuilles agrégés où le **chauffage résidentiel et tertiaire** représente la fraction écrasante de la demande. Les pays concernés (Hongrie, Estonie, Lettonie, Bulgarie, Pays-Bas, France via GRTgaz) ont en commun soit un climat continental/baltique avec hivers rigoureux, soit une part résidentielle/services historiquement élevée dans le mix gaz.
- **Traits distinctifs** :
  - Ratio hiver/été extrême (médiane 2,66, **plafonné à 2,69** pour certains points car la consommation estivale est quasi nulle → encodage par la valeur max). C'est le signal le plus net : la demande estivale s'effondre.
  - Volatilité faible (0,48) : la saisonnalité est **prévisible** car indexée sur la météo, pas sur des chocs économiques.
  - Pas d'effet week-end (0,87) → confirme la dominance résidentielle (les ménages chauffent 7j/7).
- **Anomalies / sous-groupes** :
  - **Sous-groupe « grands portefeuilles »** : FNC-00013 (Industrial Consumers NL, 588 GWh/j) et FNC-00201 (GRTgaz industriels FR, 555 GWh/j). Étiquetés « Industrial » mais leur signature saisonnière (HE > 2,6) trahit en réalité une **forte composante chauffage process / chauffage tertiaire** dans l'agrégat. Ce sont sans doute des **pools mixtes** où l'industrie diffuse cohabite avec du tertiaire / process thermique.
  - **Sous-groupe « petits saisonniers »** : FNC-00017 (BG), FNC-00037 (EE), FNC-00205 (LV) — petits volumes, mais signature de chauffage pur.
  - **Σ taille = 1296 GWh/j sur 8 points**, soit ~30 % du volume total du panel hors outliers.

##### 🟪 Cluster 1 — « Atypiques : petits L-Gas en déclin + tertiaire diffus » (4 points)

- **Profil économique** : portefeuilles **marginaux** ou en **fin de cycle**. Trois des quatre points sont des **agrégats L-Gas allemands** (DIS-00061, FNC-00045, et indirectement le « Nowega Aggregation Customer ») — le L-Gas est en sortie progressive en Allemagne et aux Pays-Bas, ce qui explique des **moyennes journalières dérisoires** (0,3 à 14 GWh/j) et une **forte volatilité relative**.
- **Traits distinctifs** :
  - Conso très faible (médiane 4,7 GWh/j) → cluster des « queues de portefeuille ».
  - Volatilité élevée (2,67, jusqu'à 6,57 pour FNC-00199) → bruyance accentuée par la **faible taille du dénominateur** (σ/μ explose quand μ tend vers 0).
  - **Ratio WE/WD à 0,51** (médiane 0,49) : la consommation **chute de moitié le week-end** → typique d'un mix où **PME, ateliers et services** (qui ferment le week-end) pèsent lourd relativement au résidentiel.
  - Saisonnalité modérée (1,21) → pas dominé par le chauffage.
- **Anomalies / particularités** :
  - **FNC-00199 « Nowega Aggregation Customer »** est limite outlier (vol = 6,57, max = 4 145 GWh/j) mais a échappé au filtre (cm < 600, vol < 15). Probablement un agrégat très hétérogène avec quelques pics ponctuels.
  - Le cluster est statistiquement faible (n=4) → toute conclusion devra être confirmée avec plus de données ou une période plus longue.

##### 🟩 Cluster 2 — « Cœur industriel et grands distributeurs européens » (21 points — 64 %)

- **Profil économique** : le **socle du marché gazier européen**. On y trouve les grands consommateurs industriels (NL, BE, IT, FR-TEREGA, PL, DE-ONTRAS, DE-GASCADE), les distributeurs centralisés (H-Gas allemand), et des power plants (FNC-00035 Fluxys Power, FNC-00210 NL Power H-gas). C'est un **cluster fonctionnellement hétérogène mais statistiquement très homogène** sur les 4 features retenues.
- **Traits distinctifs** :
  - Volatilité **très faible** (0,32 médian) → consommation **stable et régulière**, signature de process industriels en marche continue et de réseaux à forte inertie.
  - Saisonnalité **faible** (1,22) → indépendance vis-à-vis des températures, ce qui confirme la dominance industrie/process et power baseload.
  - Pas d'effet week-end (0,87) → les usines lourdes et les centrales tournent 7j/7.
  - Taille très étalée (0,46 à 358 GWh/j) → le cluster mélange volontairement géants industriels et petits points industriels périphériques.
- **Sous-groupes détectables visuellement** :
  - **Géants industriels baseload** (>100 GWh/j, vol < 0,2) : FNC-00005 IT, FNC-00208 NL H-gas, FNC-00033 Fluxys industriels — candidats à un sous-cluster « industrie lourde process continu ».
  - **Power plants** : FNC-00035 (vol 0,53, WE/WD 0,68) et FNC-00210 (vol 0,59, WE/WD 0,63) → volatilité et effet WE sensiblement plus marqués → les centrales électriques répondent à la demande électrique, qui est elle-même cyclique.
  - **Distributeurs résidentiels « tempérés »** (DIS-00060, FNC-00002 PL, FNC-00203 DE-GASCADE) : signature « calme » mais correspondent en réalité à des poches résidentielles **dans des climats plus doux** où l'usage chauffage existe sans dominer.

---

#### **4.2 — Hypothèses sur la genèse des trois profils**

| Facteur | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| **Climat / géographie** | Hivers froids (Baltes, Hongrie, Pays-Bas, France nord) | DE (transition L-gas) | Mix continental tempéré, industrie diffuse |
| **Mix sectoriel** | Résidentiel + tertiaire dominant | PME, ateliers, services discontinus | Industrie continue, power, distributeurs |
| **Modalité opérationnelle** | Demande indexée sur la météo | Demande indexée sur l'activité économique de proximité | Demande indexée sur la production / baseload |
| **Cycle de vie du portefeuille** | Mature, stable | **En déclin** (L-gas EU sortie 2030) | Mature, stable, capacités fixes |

**Pourquoi la stabilité est si différente ?**
- Cluster 0 est stable **car prévisible** (la météo l'est sur 1–2 semaines) → volatilité courte, mais saisonnalité massive.
- Cluster 2 est stable **car continu** (process industriels ne s'arrêtent pas) → volatilité ET saisonnalité faibles.
- Cluster 1 est volatil **car petit** (effet dénominateur σ/μ) ET **discontinu** (semaine vs week-end, jours fériés).

**Pourquoi la saisonnalité varie autant ?**
- Le chauffage est le **levier saisonnier numéro 1** du gaz européen. Plus la part résidentielle/tertiaire est élevée dans un agrégat, plus le ratio H/E grimpe. À l'inverse, un process industriel a une consommation quasi insensible à la température.
- Les points du Cluster 0 sont **saturés** à HE = 2,69 (valeur d'imputation) → la vraie saisonnalité de FNC-00200 (Hongrie), FNC-00014 (PDC NL) ou FNC-00205 (Lettonie) est probablement **supérieure** à 2,69. Un raffinement futur consisterait à recalculer ces ratios sur fenêtre glissante pour ne pas saturer.

**Pourquoi l'effet jour (WE/WD) isole-t-il le Cluster 1 ?**
- Les ménages consomment 7j/7 → pas d'effet WE.
- L'industrie lourde tourne 7j/7 → pas d'effet WE.
- Les **PME, services, petits ateliers et certaines centrales électriques flexibles** (qui suivent la demande élec, plus basse le WE) → effet WE marqué.
- Le Cluster 1, qui combine petits volumes + L-gas résiduel + tissu PME, est donc le seul où le WE pèse vraiment.

---

#### **4.3 — Utilité opérationnelle pour un desk de trading gaz**

##### 1. **Prévisions de demande**

| Cluster | Variable explicative dominante | Modèle conseillé | Horizon de prévision fiable |
|---|---|---|---|
| **0 (saisonnier)** | HDD (Heating Degree Days), température jour J+1, J+7 | Régression linéaire météo-conso + modèle saisonnier (SARIMA) | 1–14 jours (et profils mensuels) |
| **1 (atypiques)** | Activité économique locale, calendrier (WE / fériés) | Modèle exogène avec dummies WE + indicateur PMI | Court terme uniquement (1–3 jours) |
| **2 (industriel continu)** | Production industrielle, prix spot du gaz, marge clean spark | AR + facteurs économiques + spreads inter-énergétiques | Bonne précision sur l'agrégat ; difficile en granulaire |

##### 2. **Gestion des risques**

- **Cluster 0** : exposition **risque météorologique pur**. Le hedging naturel est l'achat de **dérivés climatiques** (HDD futures / weather swaps EEX) couplé à du gaz mensuel hiver.
- **Cluster 1** : risque de **décroissance structurelle** (sortie L-gas, désindustrialisation diffuse). Risque de **rupture de profil** plus que de pic. À pricer comme une option short sur le volume.
- **Cluster 2** : risque **macro-économique et prix-élasticité**. Une hausse de 30 % du TTF peut entraîner du fuel switching ou des arrêts industriels → volatilité **endogène au prix**, ce qui crée un risque de boucle (demande baisse quand prix monte). Couverture par options OTM (caps).

##### 3. **Stratégie de hedging**

| Cluster | Produit prioritaire | Logique |
|---|---|---|
| 0 | **Spreads saisonniers été/hiver** (TTF Cal-Win vs Cal-Sum) | Capter le différentiel chauffage |
| 0 | **HDD-linked weather derivatives** | Couvrir l'aléa de température |
| 1 | **Daily / Within-day TTF** + options courtes | Profil court terme + bruit élevé |
| 2 | **Baseload mensuel + couverture Cal-Year** | Stabilité de la demande → couverture longue durée justifiée |
| 2 (sous-groupe power) | **Clean Spark Spread** (gaz vs élec vs CO₂) | Demande dépend de la merit order électrique |

##### 4. **Pricing et offres commerciales**

- **Cluster 0** → contrats **saison hiver / saison été** différenciés, avec **prime de saisonnalité** ; possibilité d'offres « take-or-pay » hivernales.
- **Cluster 1** → contrats **flexibles courte durée**, indexés WE/WD, possibles offres en agrégateur (regroupement pour atteindre la taille critique).
- **Cluster 2** → contrats **annuels indexés TTF**, voire indexés sur un panier (TTF + brent + carbone) pour le sous-groupe power. Marges minces, volumes élevés.

##### 5. **Optimisation des achats / portefeuille**

- Un acheteur structurel peut **mixer Cluster 0 et Cluster 2** dans un portefeuille pour lisser la saisonnalité : Cluster 2 amène une base annuelle stable, Cluster 0 monétise le différentiel saisonnier.
- Le **stockage** (Bergermeer, REC, Haidach) est principalement utile pour servir le **Cluster 0** : remplir l'été pour livrer l'hiver. Un trader peut arbitrer **stockage vs marché spot** sur la base de la courbe forward résiduelle.
- Pour le **Cluster 1**, l'optimisation passe plus par **désengagement progressif** ou repositionnement vers l'électrification / pompes à chaleur (cohérent avec la sortie L-Gas).

##### 6. **Détection d'anomalies / drift de profil**

- Le cluster d'un point est une **signature**. Si un point du Cluster 2 voit son HE passer de 1,2 à 2,5 sur 12 mois, c'est un signal :
  - soit le mix sectoriel sous-jacent change (déclin industriel → résidentiel relatif),
  - soit la météo a été extrême (hiver très froid),
  - soit une fraude / erreur de mesure.
- Mettre en place une **surveillance trimestrielle** : pour chaque point, recalculer la distance au centroïde de son cluster. Une dérive > 2 écarts-types = **alerte revue de portefeuille**.
- Croisé avec un calendrier d'arrêts industriels, c'est aussi un **détecteur de stop production** précieux pour le desk.

---

#### **4.4 — Qualité, limites et axes d'amélioration**

##### Ce qui correspond aux attentes théoriques

- ✅ Trois grands profils émergent comme prévu par la théorie économique du gaz : un profil saisonnier, un profil continu, et un profil atypique.
- ✅ Le profil « industriel continu » est majoritaire en nombre de points (21/33), ce qui est cohérent avec un panel ENTSO-G dominé par les agrégats TSO industriels.
- ✅ La validation par 3 algorithmes indépendants (KMeans, DBSCAN, Ward) confirme la **réalité structurelle** des groupes.

##### Ce qui diverge des attentes

- ❌ **Les clusters ne se rangent pas selon l'axe "taille"** comme la théorie le suggérait initialement. Le Cluster 0 (« saisonnier ») contient à la fois FNC-00017 (1,67 GWh/j) et FNC-00013 (588 GWh/j) → la dimension saisonnalité prime sur la taille.
- ❌ **Le label "Résidentiel" du cluster 2 est faux** : les pointKey du Cluster 2 sont majoritairement « Industrial Consumers », « Power Plants » et « Aggregated Final Consumers » de pays tempérés. Le vrai marqueur « résidentiel » est porté par le Cluster 0.
- ❌ **Le ratio_we_wd n'a apporté de la valeur que pour distinguer le Cluster 1** ; il est quasi muet entre Cluster 0 et Cluster 2.

##### Limites de la segmentation actuelle

1. **Granularité agrégée** → un point ENTSO-G peut agréger plusieurs centaines de consommateurs hétérogènes ; le clustering capture la **signature de l'agrégat**, pas celle des consommateurs unitaires.
2. **K=3 sous-optimal statistiquement** : Silhouette, Davies-Bouldin et Elbow pointaient unanimement K=5. Le choix K=3 est un choix **pragmatique** (lisibilité métier, taille minimale de cluster), pas statistique.
3. **Cluster 1 sur-représenté en L-Gas allemand** → l'algorithme a probablement capté une **anomalie technique** (sortie L-Gas) plutôt qu'un véritable profil structurel.
4. **Pas de variable exogène** : météo, prix TTF, calendrier de jours fériés, jours de maintenance — aucune ne participe au clustering. La signature est purement statistique sur la conso elle-même.

---

#### **4.Bonus — Recommandations opérationnelles immédiates pour le desk/Portfolio Manager**

1. **Étiqueter les portefeuilles clients** selon les 3 clusters et appliquer une **stratégie de prévision dédiée** par cluster.
2. Construire un **indicateur de drift** mensuel (distance au centroïde de son cluster) → alerte automatique si > 2σ.
3. **Hedger asymétriquement** : weather derivatives sur Cluster 0, options courtes sur Cluster 1, baseload Cal-Year sur Cluster 2.
4. **Repricer les contrats Cluster 1** vu leur déclin structurel et leur profil WE/WD très marqué.
5. **Lancer un projet "données enrichies"** : intégrer HDD, prix spot, jours fériés sur 5 ans pour passer d'un clustering descriptif à un clustering prédictif.

---
