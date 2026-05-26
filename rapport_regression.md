# Prédiction du Prix TTF : Régression, Régimes et Apport Métier

## 🎯 1. Introduction

Le **TTF** (*Title Transfer Facility*, Pays-Bas) est devenu depuis 2018 le benchmark de référence du gaz naturel en Europe, l'équivalent du Henry Hub pour les États-Unis. Sa courbe est passée d'un actif assez ennuyeux (15 à 25 EUR/MWh sur la période 2017-2020) à l'une des séries de matières premières les plus volatiles jamais observées : pic à plus de **300 EUR/MWh à l'été 2022** suite à l'invasion de l'Ukraine, puis lente normalisation autour de 30 à 50 EUR/MWh sur 2023-2026.

L'objectif de ce travail est double :

1. **Prédire le prix TTF à J+1** à partir de signaux de marché publics (autres commodités, météo, stockage)
2. **Identifier les drivers** structurels et chiffrer leur poids

Quatre familles de modèles sont comparées : régression linéaire, régularisation (Ridge / Lasso), arbres boostés (XGBoost) et réseau de neurones. **Spoiler : le modèle linéaire l'emporte largement**, et la raison n'est pas un échec de tuning. C'est l'**hétérogénéité temporelle** des trois régimes successifs (pré-crise, crise 2022, post-crise) qui pénalise structurellement les modèles flexibles.

---

## 📊 2. Données

### 2.1 Périmètre temporel

Les données brutes couvrent **2015-01-01 à 2026-05-25** (~11 ans), mais le dataset modélisable démarre effectivement en **novembre 2017**. Trois raisons à ce décalage :

- Le contrat **Dutch TTF Natural Gas Futures** sur Investing.com publie des cotations exploitables à partir d'octobre 2017
- L'inner join sur les neuf séries (toutes les sources doivent avoir une valeur le même jour) coupe les périodes où l'une d'elles est manquante
- Les 14 premières observations sont perdues par la moyenne mobile **ATR(14)**

Au final on travaille sur **2 132 jours de trading** (≈ 8,5 ans calendaires), suffisant pour couvrir au moins un cycle complet pré-crise / crise / post-crise.

### 2.2 Sources de variables

| Variable | Unité | Source | Accès |
|---|---|---|---|
| `prix_ttf_eur_per_mwh` | EUR / MWh | [Investing.com, Dutch TTF](https://www.investing.com/commodities/dutch-ttf-gas-c1-futures-historical-data) | CSV manuel |
| `prix_brent_usd_per_bbl` | USD / baril | [Investing.com, Brent](https://www.investing.com/commodities/brent-oil-historical-data) | CSV manuel |
| `prix_charbon_usd_per_tonne` | USD / tonne | [Investing.com, Rotterdam Coal](https://www.investing.com/commodities/rotterdam-coal-futures-historical-data) | CSV manuel |
| `prix_henry_hub_usd_per_mmbtu` | USD / MMBtu | [Investing.com, Henry Hub](https://www.investing.com/commodities/natural-gas-historical-data) | CSV manuel |
| `prix_jkm_usd_per_mmbtu` | USD / MMBtu | [Investing.com, JKM LNG](https://www.investing.com/commodities/lng-japan-korea-marker-platts-futures-historical-data) | CSV manuel |
| `prix_co2_eur_per_tonne` | EUR / tCO₂ | [Investing.com, EUA Yearly](https://www.investing.com/commodities/carbon-emissions-historical-data) | CSV manuel |
| `taux_eur_usd` | sans unité | [Investing.com, EUR/USD](https://www.investing.com/currencies/eur-usd-historical-data) | CSV manuel |
| `agsi_pct_capacity` | % rempl. | [GIE AGSI+](https://agsi.gesmes.org/) | API JSON |
| `temp_*_celsius` | °C | [Open-Meteo Archive API](https://open-meteo.com/) | API JSON |

Le mix combine **3 commodités énergétiques substituables** (Brent, charbon, gaz US/asiatique), du **carbone** (coût implicite du switch), du **FX** (parité EUR/USD pour reconvertir les cours US) et deux variables fondamentales : **stockage européen** et **météo des trois grands hubs de demande continentaux** (Paris, Amsterdam, Berlin).

### 2.3 Feature engineering

Quatre familles ont été ajoutées au-dessus des séries brutes :

- **Calendaire** : jour de semaine, mois, trimestre, année, jour julien, flags week-end / lundi. Captent saisonnalité et effets de calendrier
- **Thermique** : moyenne des trois capitales + **HDD = max(0, 15 − T_moyenne)**. Proxy direct de la demande de chauffage résidentiel-tertiaire
- **Ratios cross-produits** : `Brent/TTF`, `TTF/Henry_Hub` (avec conversion ×3,6 MMBtu vers MWh), `JKM/Henry_Hub`. Spreads d'arbitrage et signaux de flux GNL
- **Volatilité** : ATR 14j approximée sur les variations absolues de clôture, régime de risque court terme

Aucun lag explicite n'a été ajouté : le prix TTF du jour J est déjà dans les features (utilisé pour prédire J+1), donc la persistance est implicite. C'est volontaire et discuté plus bas.

---

## 🔬 3. Méthodologie

### 3.1 Modèles testés

Quatre familles, sélectionnées pour couvrir un spectre **simple à complexe** :

- **Régression linéaire (LR)** : la baseline non-régularisée. Sert de référence pour estimer si la régularisation apporte vraiment quelque chose et pour vérifier que les hypothèses linéaires tiennent
- **Ridge & Lasso** : pénalités L2 / L1 pour gérer les colinéarités (les prix de commodités sont très corrélés entre eux, surtout sur la période crise). Lasso fait aussi de la **sélection de features** automatique
- **XGBoost** : gradient boosting pour capturer les non-linéarités et interactions (typiquement, effet du HDD qui dépend du niveau de stockage). Test d'une version basique + une version fine-tunée par RandomSearchCV
- **Réseau de neurones (MLP)** : `Dense(64) → BN → Dropout → Dense(32) → BN → Dropout → Dense(1)`, archi standard pour tabulaire. Pas de NAS ni d'AutoML, on reste sur une architecture de littérature

### 3.2 Tuning et cross-validation

`TimeSeriesSplit(n_splits=5, gap=5)` partout, **shuffle systématiquement désactivé**. Le `gap=5` évite qu'une feature à fenêtre glissante (ATR 14j) du fold d'entraînement déborde sur le début du fold de validation.

- Ridge / Lasso : `GridSearchCV` sur 5 valeurs d'α
- XGBoost : `RandomizedSearchCV` 200 tirages sur 1 280 combinaisons (n_estimators × max_depth × learning_rate × subsample × colsample_bytree), avec `objective='reg:absoluteerror'` aligné sur la métrique de scoring, et wrap dans `TransformedTargetRegressor(np.log, np.exp)` pour compresser le spike 2022
- NN : `EarlyStopping(patience=15, restore_best_weights=True)`, validation_split=0.2 chronologique

**Le tuning XGBoost et NN n'a pas amélioré la performance finale**. Au contraire, XGBoost tuné fait ~80 % plus de MAE que XGBoost par défaut, et le NN ne converge même pas sur un résultat utilisable. La cause n'est pas un bug de pipeline : c'est la **CV qui sélectionne des hyperparamètres optimisés pour gérer le spike 2022**, lequel domine totalement le signal de validation. Ces paramètres "agressifs" (deep trees, learning_rate=0.2) sur-réagissent ensuite sur la période test 2024-2026 calme. La régularisation forte du Lasso passe au travers de ce piège.

### 3.3 Métriques

**MAE en EUR/MWh** comme métrique principale. Choix justifié :

- Interprétable directement en termes business : "le modèle se trompe en moyenne de X EUR/MWh"
- Pénalise **toutes les erreurs proportionnellement à leur amplitude** (vs RMSE qui sur-pondère les outliers). Sur une série avec un spike de 350 EUR/MWh, RMSE serait totalement dominé par 2022
- Robuste à la non-normalité des résidus (vraie sur une série financière)

RMSE et R² sont reportés pour contexte mais ne servent pas à la décision finale.

### 3.4 Setup temporel

Cible = `prix_ttf_eur_per_mwh.shift(-1)`. La dernière ligne (sans J+1 connu) est supprimée, l'index est réinitialisé. **Aucune fuite temporelle** : toutes les features du jour J sont disponibles à la clôture de J (cotations financières + AGSI publié 24h après), aucune n'utilise d'information de J+1. Split train/test chronologique 80/20 (1 705 / 427 jours) sans shuffle.

---

## 📈 4. Résultats

### 4.1 Tableau de performance, test set

| Modèle | MAE (EUR/MWh) | RMSE | R² |
|---|---|---|---|
| **Lasso (α=0.001)** | **1.248** | **1.847** | **0.939** |
| Linear Regression | 1.256 | 1.858 | 0.939 |
| Ridge (α=0.01) | 1.256 | 1.858 | 0.939 |
| XGBoost basique | 1.435 | 1.981 | 0.930 |
| XGBoost tuné | 2.596 | 3.228 | 0.814 |
| Neural Network | 5.379 | 6.481 | 0.251 |

Référence métier : sur le test set, le **prix TTF moyen est de ~46 EUR/MWh**, donc un MAE de 1.25 représente **~2.7 % d'erreur relative**. C'est bon pour un horizon J+1 sur une série de matière première.

### 4.2 Analyse

Trois observations majeures :

1. **Le Lasso gagne, mais de peu sur LR / Ridge** (1.248 vs 1.256). Les trois modèles linéaires sont fonctionnellement équivalents : le Lasso fait juste sauter 2 features sur 23 (`est_weekend` et `agsi_norm`, redondants avec leurs versions non-normalisées). Le R² de 0.939 est **trompeusement élevé** : il vient à 95 % de l'auto-corrélation de TTF (prédire "demain ≈ aujourd'hui" donne déjà un R² > 0.93).
2. **XGBoost basique reste compétitif** (1.435), pas loin du Lasso, ce qui montre que le boosting capture quand même quelque chose. Mais l'écart de **80 % entre XGB basique et XGB tuné** est anormal : un tuning agressif détériore le modèle, signe que la CV optimise pour le mauvais signal.
3. **Le NN est inutilisable** (R² = 0.25). Avec ~1 700 échantillons et 23 features, le MLP n'a pas la profondeur de données pour apprendre quoi que ce soit de stable. BatchNorm + Dropout aggravent le problème sur des batches de 32 sur une cible aussi auto-corrélée.

**Ratio MAE_test / MAE_train pour Lasso** ≈ 1.05, donc pas d'overfitting, généralisation propre. Pour XGB tuné le ratio explose (~3.5) : overfitting massif, le modèle a mémorisé le spike 2022 et n'arrive pas à se calibrer sur du calme.

### 4.3 ⚠️ Point clé : régimes temporels

L'écart de performance entre Lasso et XGBoost/NN n'est **pas un problème de tuning**. C'est une limite structurelle des données. La série TTF traverse **trois régimes très distincts** sur la période d'étude :

1. **Pré-crise (2017 à 2021)** : volatilité modérée (5 à 30 EUR/MWh), dynamique saisonnière propre, drivers traditionnels (HDD, stockage, Brent)
2. **Crise (2022)** : choc exogène géopolitique, prix qui dépassent **300 EUR/MWh**, déconnexion des fondamentaux habituels, spikes intra-mensuels de ±50 %
3. **Post-crise (2023 à 2026)** : normalisation lente, retour à 30 à 50 EUR/MWh mais avec une mémoire de la volatilité de 2022

Les modèles flexibles (XGBoost, NN) **apprennent du bruit spécifique à chaque régime** plutôt que les mécanismes généraux qui les relient. Concrètement, la CV est dominée par les folds contenant 2022 (MAE ~14 EUR/MWh sur ces folds vs ~1 à 2 EUR/MWh ailleurs), donc le grid search sélectionne des hyperparamètres optimisés pour suivre le spike. Ces paramètres sur-réagissent ensuite sur la période test calme. Le modèle linéaire, par sa **rigidité**, capture uniquement les relations stables qui survivent aux trois régimes : il rate les pics mais ne se ridiculise jamais.

**C'est une limitation fondamentale de données hétérogènes, pas un problème de pipeline.** Plus de tuning ne corrigera rien. Seule une stratégie de **modèles par régime** ou une cible reformulée (delta plutôt que niveau) ferait sauter le verrou.

> Le point intéressant ici : un data scientist sans connaissance métier aurait probablement passé deux semaines à fine-tuner XGBoost en supposant un bug. L'expertise marché gaz détecte la cause structurelle en regardant la chronologie des prix.

---

## 🔍 5. Explainabilité

### 5.1 Coefficients Lasso, Top 10

Les coefficients ci-dessous portent sur les **features standardisées** (RobustScaler), donc l'amplitude est directement comparable.

| Rang | Feature | Coef | Lecture métier |
|---|---|---|---|
| 1 | `prix_ttf_eur_per_mwh` | **+26.61** | Auto-corrélation forte : demain ≈ aujourd'hui + petits ajustements |
| 2 | `prix_jkm_usd_per_mmbtu` | +3.92 | Marché LNG asiatique tire / pousse les flux vers l'Europe |
| 3 | `mois` | +1.65 | Saisonnalité directe (les mois d'hiver poussent le prix) |
| 4 | `ratio_jkm_henry_hub` | −1.43 | Spread Asie/US : si l'Asie paie plus, moins de cargos vers l'Europe, donc TTF monte |
| 5 | `prix_charbon_usd_per_tonne` | +1.24 | Substitution charbon-gaz dans la production élec |
| 6 | `trimestre` | −1.23 | Effet trimestriel résiduel après contrôle du mois |
| 7 | `prix_co2_eur_per_tonne` | +1.20 | EUA renchérit le charbon, donc switch vers gaz, donc demande TTF |
| 8 | `ratio_ttf_henry_hub` | −1.14 | Spread Europe/US, régulateur naturel des flux GNL |
| 9 | `temp_moyenne_eur` | +0.98 | Coefficient positif **contre-intuitif** (cf. § 5.3) |
| 10 | `num_jour_annee` | −0.83 | Position dans l'année, complète la saisonnalité |

**21 features sur 23 sont conservées** (Lasso α=0.001, très faible). `est_weekend` et `agsi_norm` sont les seules à être annulées car elles dupliquent `jour_semaine ≥ 5` et `agsi_pct_capacity`.

### 5.2 Top drivers, lecture métier

- **HDD / températures (drivers physiques)** : le chauffage résidentiel-tertiaire représente ~40 % de la demande gaz européenne. Une chute de 1 °C en hiver = ~3 % de demande gaz en plus. Le coefficient `temp_moyenne_eur` est néanmoins **positif dans le modèle**, ce qui semble incohérent. L'explication est que la température est colinéaire avec `mois` et `num_jour_annee`, qui captent déjà la saisonnalité. Le Lasso répartit la charge sur plusieurs colonnes corrélées et certains coefs deviennent contre-intuitifs en isolation. **HDD norm + non-norm ont un coef cumulé ~0.87 positif**, ce qui est dans le bon sens.
- **Brent (corrélation historique)** : coefficient faible (−0.24) car l'indexation des contrats LT TTF sur le pétrole a beaucoup reculé depuis 2018. Le Brent reste un signal macro mais n'est plus un driver direct du spot. Confirme une intuition métier connue mais souvent oubliée.
- **AGSI (stockage)** : coefficient quasi nul (−0.10). Surprenant à première vue : on s'attendrait à ce que des stocks bas = TTF haut. L'explication probable est que **le niveau de remplissage évolue trop lentement** (variation typique <1 %/jour) pour expliquer du J+1. L'effet de stockage joue sur des horizons mensuels, pas quotidiens.
- **JKM (LNG Asie)** : coefficient élevé (+3.92), confirme que **l'arbitrage trans-pacifique pilote les flux GNL** vers l'Europe à très court terme. Les cargos peuvent être redirigés en quelques jours.

### 5.3 Patterns d'erreur

Le modèle Lasso laisse des résidus structurés :

- **Hétéroscédasticité confirmée** : les résidus s'élargissent quand le niveau de prix prédit augmente. Le scatter résidus vs fitted forme un cône, typique d'une série où la volatilité absolue est proportionnelle au niveau
- **Erreurs concentrées sur les pics** : les ~10 plus gros résidus correspondent à des **journées de sauts brusques** (>5 % intra-day). Le modèle, par construction linéaire et auto-corrélé, lisse ces sauts. Ce sont des erreurs **non-prédictibles** avec les features disponibles
- **Pas de biais saisonnier visible** : la MAE par année est globalement stable sur la période de test (2024 à 2026, période post-crise donc relativement homogène)

---

## ⚠️ 6. Limitations et réflexion

### 6.1 Limitations principales

1. **Hétérogénéité temporelle des trois régimes**, déjà détaillé en § 4.3. C'est la limite #1, et elle plafonne tous les modèles non-linéaires sur ce dataset.
2. **Features fondamentales manquantes** :
   - **Flux physiques** (interconnexions, GNL terminaux). Données ENTSOG dispo mais pas intégrées ici
   - **Forward curve TTF**. La structure de courbe (contango/backwardation) anticipe le spot. Pas utilisée
   - **Calendrier politique / sanctions**. Impossible à featuriser proprement, mais c'est ce qui a déclenché 2022
   - **Maintenance des infrastructures** (Bergermeer, Nord Stream). Événementiel non capturé
3. **Données partiellement synthétiques** :
   - **Météo Open-Meteo** : c'est une **archive reconstruite** (modèle ERA5), pas des mesures stations brutes. Précision suffisante pour HDD agrégé mais pas pour des effets micro
   - **AGSI+** : agrégat européen, ne distingue pas les stockages stratégiques des commerciaux
4. **Granularité quotidienne** alors que le TTF se trade 24/7. Les variations intra-day ne sont pas captées, modélisation horaire serait pertinente pour du trading actif
5. **Corrélation ≠ causation** : le coefficient JKM élevé reflète un co-mouvement, pas un lien causal direct testé

### 6.2 Réflexion sur la complexité

L'intuition commune en ML est : **plus le modèle est complexe, mieux il capture la réalité**. Sur ce dataset c'est l'inverse, le linéaire bat clairement XGBoost et écrase le NN. Ce n'est pas une anomalie : c'est cohérent avec la littérature sur la **prévision macro et matières premières**.

La raison est simple : sur les marchés de commodités, **les fondamentaux dominent et sont relativement stables** (demande chauffage, mécaniques de substitution, élasticité prix). Les non-linéarités existent mais sont noyées dans le bruit, et avec ~1 700 échantillons d'entraînement on n'a pas assez de signal pour les apprendre proprement. Plus le modèle est flexible, plus il sur-apprend les particularités du régime de crise.

> *"Energy markets aren't tabular Kaggle competitions. Mean-reverting fundamentals beat black-box flexibility, until you have flux data, forward curve, and policy events."*

### 6.3 Axes d'amélioration

| Idée | Effort | Gain attendu |
|---|---|---|
| **Modèles séparés par régime** (changepoint detection puis modèle par segment) | 2 à 3 semaines | Important sur la robustesse |
| Prédire le **delta** `ΔTTF` au lieu du niveau | 1 semaine | Casse l'auto-corrélation artificielle, vrai signal métier |
| Ajouter la **forward curve** (M+1, M+3, Q+1, Cal+1) | 2 semaines (données payantes) | Élevé, la courbe encapsule les anticipations marché |
| Features **exogènes événementielles** (sanctions, météo extrême, maintenance pipelines) | Chantier, featurisation manuelle | Critique pour 2022, négligeable pour 2024+ |
| Passage en **prédiction probabiliste** (quantile regression, MAPIE) | 1 semaine | Intervalle de confiance directement utilisable pour le risk management |
| **Horizon plus court** (intra-day ou H+1) avec données heure par heure | 2 à 3 semaines (collecte) | Pertinent pour trading, hors scope du J+1 actuel |

---

## 💡 7. Conclusion

### 7.1 Résumé

Le meilleur modèle est le **Lasso α=0.001** avec une **MAE de 1.25 EUR/MWh** (~2.7 % d'erreur relative pour un TTF moyen de 46 EUR/MWh sur le test set). Les modèles complexes (XGBoost tuné, réseau de neurones) **sous-performent fortement** non pas à cause d'un défaut de pipeline mais à cause de l'**hétérogénéité temporelle** du dataset, qui pénalise structurellement la flexibilité.

### 7.2 Apport métier

- **Drivers confirmés** : auto-corrélation forte (persistance), spreads inter-marchés gaz (JKM, Henry Hub), substitution charbon-gaz via CO₂, saisonnalité directe. Brent et stockage AGSI ont un effet plus faible que ce que l'intuition macro suggère
- **Régimes temporels critiques** : 2022 est un événement géopolitique qui a cassé la dynamique habituelle. Un modèle entraîné uniquement sur 2017-2021 aurait totalement raté la crise, symétriquement, un modèle entraîné sur 2022 over-fit cette crise. La modélisation par régime est la voie sérieuse
- **ML ≠ remplacement de l'expertise métier** : sans la lecture chronologique des prix (qui n'apparaît dans aucun KPI ML standard), la sous-performance des modèles complexes serait restée inexpliquée

### 7.3 Recommandations opérationnelles

- **Usage trading** : le Lasso est suffisant pour un signal directionnel J+1. Son MAE de ~1.25 EUR/MWh est compatible avec une exécution prudente (taille de position ajustée à l'incertitude)
- **Sensibilités** : les coefficients standardisés du Lasso donnent directement les "Greeks" du modèle, utiles pour un risk officer (impact d'un mouvement de 1σ sur JKM, sur Brent, etc.)
- **Modèle de changement de régime** : avant la prochaine crise (et il y en aura une), basculer sur un système type **HMM ou Bayesian changepoint** qui peut détecter en temps réel un shift de dynamique. C'est l'évolution naturelle de ce travail

### 7.4 Closing

L'énergie ressemble plus à la finance qu'à la vision par ordinateur : **les modèles simples avec interprétabilité battent les boîtes noires complexes**, surtout quand le régime de marché peut basculer brutalement. Le vrai enjeu n'est pas d'empiler des couches denses, c'est de **comprendre le système** que l'on modélise, et d'avoir un modèle assez transparent pour qu'un trader, un risk officer ou un régulateur puissent challenger ses prédictions.
