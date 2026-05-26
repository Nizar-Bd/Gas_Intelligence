# GasIntelligence Hub, Portfolio Data Lab

Data lab appliqué au marché du gaz naturel européen : segmentation des consommateurs, classification de messages REMIT, prédiction du prix TTF.

## 📌 Contexte

Le marché du gaz naturel européen a connu depuis 2022 une volatilité sans précédent. Le TTF (Title Transfer Facility), benchmark continental, est passé de ~20 EUR/MWh en 2019 à plus de 300 EUR/MWh en 2022, avant de se stabiliser autour de 30-50 EUR/MWh depuis 2023.

Ce portfolio explore trois questions complémentaires sur ce marché à travers trois projets data indépendants, articulés autour d'un même socle de données.

 L'objectif : démontrer une chaîne complète de raisonnement data, de la récupération des données à l'interprétation métier.

## 🧭 Trois axes d'analyse

### Axe 1 : Segmentation des profils de consommation

Identifier les typologies naturelles de consommateurs gaz à partir des séries quotidiennes ENTSOG. Approche hybride combinant exploration descriptive, KMeans, DBSCAN et clustering hiérarchique pour valider la robustesse des groupes.

[Voir le rapport →](./rapport_clustering.md)

### Axe 2 : Classification NLP des messages REMIT

Classifier automatiquement les notifications REMIT (Regulation on Energy Market Integrity and Transparency) par type d'incident (maintenance, panne, fuite, etc.). Approche text mining + modèle supervisé pour aider le monitoring opérationnel.

[Voir le rapport →](./rapport_nlp.md)

### Axe 3 : Régression et prédiction du prix TTF

Prédire le prix TTF à J+1 à partir de signaux de marché publics (autres commodités, météo, stockage). Comparaison Lasso / Ridge / XGBoost / réseau de neurones, avec une analyse des limites liées à l'hétérogénéité temporelle 2017-2026.

[Voir le rapport →](./rapport_regression.md)

## Données
- [Investing.com](https://www.investing.com/) : cotations commodités historiques (CSV)
- [GIE AGSI+](https://agsi.gesmes.org/) : taux de remplissage des stockages européens (API)
- [Open-Meteo Archive](https://open-meteo.com/) : températures quotidiennes ERA5 (API)
- [ENTSOG](https://www.entsog.eu/) : flux et consommation gaz Europe
- REMIT : messages mockés à partir du schéma officiel

## 📚 Accès direct aux rapports

| Axe | Rapport | Notebook |
|---|---|---|
| Segmentation | [rapport_clustering.md](./rapport_clustering.md) | [1.clustering_patterns_consommation.ipynb](./notebooks/1.clustering_patterns_consommation.ipynb) |
| NLP / Classification | [rapport_nlp.md](./rapport_nlp.md) | [2.nlp_network_messages.ipynb](./notebooks/2.nlp_network_messages.ipynb) |
| Régression TTF | [rapport_regression.md](./rapport_regression.md) | [3.prediction_prix_ttf_regression.ipynb](./notebooks/3.prediction_prix_ttf_regression.ipynb) |
