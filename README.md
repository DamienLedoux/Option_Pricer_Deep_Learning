# Pricer-par-MLP-et-extracteur-de-Greeks-par-diff-rentiation-automatique
Il s'agit d'un projet étudiant, réalisé dans le cadre du cours d'apprentissage profond de l'école Centrale de Lille. L'idée pour nous était d'appréhender les outils de deep learning d'une manière originale : utiliser les données de Kaggle pour entraîner un modèle qui nous servira de pricer.

Notebook d'un pricer d'options par apprentissage : un réseau de neurones (MLP) est entraîné à approximer le prix d'options à partir de prix de marché réellement observés (SPY, 2020-2021), puis comparé à Black-Scholes sur une période jamais vue à l'entraînement (2022). Les Greeks du réseau (Delta, Gamma, Vega) sont extraits par différentiation automatique via `tf.GradientTape`, et le smile de volatilité implicite est reconstruit à partir du modèle.

## Dépendances

```bash
pip install numpy pandas matplotlib scipy scikit-learn tensorflow requests
```

## Données

Le notebook utilise la base d'options quotidiennes du S&P 500 (SPY) sur 2020-2022, à télécharger sur Kaggle :

https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022

Le taux sans risque (bons du Trésor américain à 3 mois) est récupéré automatiquement via l'API FRED (une connexion internet suffit, aucune clé requise).

## Utilisation

Ouvrir `Projet_APR.ipynb` dans Jupyter, indiquer le chemin du CSV Kaggle dans la partie « Imports et hyperparamètres », puis exécuter les cellules dans l'ordre.

## Contenu

Préparation et nettoyage des données réelles, construction et entraînement du MLP (Keras), comparaison tripartite MLP / Black-Scholes / marché, métriques d'erreur (MAE, RMSE, MAPE, R2), extraction des Greeks par différentiation automatique, reconstruction du smile de volatilité implicite, et bilan critique des résultats.

Réalisé par Kim TRINH et Damien LEDOUX.
