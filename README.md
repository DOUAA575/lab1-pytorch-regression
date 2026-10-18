# Lab 1 : Régression avec un DNN/MLP en PyTorch (dataset NYSE)

**Objectif :** prendre en main PyTorch pour résoudre une tâche de régression avec un réseau de neurones de type MLP.
**Problème choisi :** prédire le prix de clôture (`close`) d'une action à partir de `open`, `low`, `high` et `volume`.

**Dataset :** [NYSE sur Kaggle](https://www.kaggle.com/datasets/dgawlik/nyse), fichier `prices-split-adjusted.csv` (environ 851 000 lignes, 501 actions, de 2010 à 2016). J'ai pris ce fichier car les prix y sont corrigés des splits d'actions.

**Outils :** Kaggle, PyTorch, scikit-learn, skorch, pandas, matplotlib, seaborn.
Les graphes et les sorties détaillées sont dans le notebook `notebook_regression.ipynb`.

---

## 1. Analyse exploratoire (EDA)

J'ai regardé la structure des données (valeurs manquantes, doublons, statistiques), puis tracé les histogrammes, la matrice de corrélation, les boxplots, l'évolution de quelques actions et le nuage `open` / `close`.

- Les prix ont des distributions quasi identiques, très asymétriques à droite : la plupart des valeurs sont entre 0 et 100 $, avec une longue queue jusqu'à environ 1 600 $.
- Le volume est encore plus écrasé près de zéro (maximum d'environ 8,5 × 10⁸).
- `open`, `high`, `low` et `close` sont corrélés à presque 1, et la relation `open` / `close` est quasi linéaire : la tâche est donc plutôt facile.
- Les échelles sont très différentes entre prix et volume, donc la **normalisation est indispensable**. Les outliers sont conservés, car ils correspondent à des situations réelles du marché.

## 2. Architecture du réseau

- Découpage : 80 % train / 20 % test.
- Normalisation avec `StandardScaler`, ajusté **uniquement sur le train** pour éviter la fuite de données (la cible est aussi normalisée).
- MLP : 4 entrées, des couches cachées `Linear + ReLU`, et 1 neurone de sortie **sans activation** (valeur continue).
- Perte : MSE. Optimiseur : choisi par le GridSearch.
- Comme l'accuracy de classification n'existe pas en régression, j'utilise le **R²** comme mesure de performance (1 = parfait).

## 3. GridSearch

Recherche avec `GridSearchCV` de scikit-learn, en enveloppant le modèle PyTorch avec `skorch`, validation croisée à 3 plis. Pour gagner du temps, la recherche a été faite sur un **sous-échantillon de 50 000 lignes** (batch de 256).

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD, RMSprop |
| Epochs | 20, 40 |
| Architecture | (32), (64, 32), (128, 64, 32) |

**Meilleure configuration :** lr = [...], optimiseur = [...], architecture = [...], 40 epochs.

L'entraînement final sur toutes les données utilise un batch de 1024 pour aller plus vite.

## 4. Courbes Loss/Epochs et Accuracy (R²)/Epochs

Pour le modèle 1 (meilleure configuration), la loss baisse vite durant les premières epochs puis se stabilise autour de 10⁻⁴, et le R² dépasse 0,9999 en train comme en test. Les courbes train et test sont presque confondues : **ni overfitting, ni underfitting**. Les petites oscillations en fin d'entraînement suggèrent un learning rate un peu élevé pour la phase finale.

## 5. Régularisation

J'ai comparé le modèle de base à cinq variantes, avec les mêmes hyperparamètres et la même initialisation :

| Modèle | Train loss | Test loss | Train R² | Test R² | Epochs |
|---|---|---|---|---|---|
| 1-Base | 0.00006 | 0.00007 | 0.99994 | 0.99993 | 40 |
| 2-L2 (weight decay) | 0.00009 | 0.00009 | 0.99991 | 0.99991 | 40 |
| 3-Dropout | 0.00088 | 0.00089 | 0.99912 | 0.99911 | 40 |
| 4-BatchNorm | 0.00804 | 0.00816 | 0.99196 | 0.99189 | 40 |
| 5-Early Stopping | 0.00010 | 0.00010 | 0.99990 | 0.99990 | 16 |
| 6-Combiné (L2 + Dropout + BatchNorm) | 0.00656 | 0.00663 | 0.99344 | 0.99341 | 40 |

**Analyse :** le modèle de base reste le meilleur. Comme l'écart train/test était déjà nul, il n'y avait pas d'overfitting à corriger. Dropout, BatchNorm et la combinaison dégradent les résultats et rendent l'entraînement instable (le R² de la combinaison descend jusqu'à 0,95), car le problème est quasi linéaire et ces techniques ajoutent du bruit inutile. L'Early Stopping est la plus intéressante : presque la même précision en 2,5 fois moins d'epochs.

## 6. Synthèse personnelle

Ce TP m'a permis de comprendre toute la chaîne d'un projet de deep learning avec PyTorch : préparation des données, construction d'un MLP, boucle d'entraînement (forward, perte, backpropagation, mise à jour des poids) et évaluation. J'ai vu l'impact du learning rate et de l'optimiseur, et l'intérêt du GridSearch, même s'il coûte cher en temps de calcul. J'ai aussi compris que la régularisation n'est pas toujours utile : quand le modèle ne sur-apprend pas, elle peut même dégrader les résultats.

**Difficultés :** le temps de calcul (le DataLoader était lent, je l'ai remplacé par un découpage direct des tenseurs), le choix d'une métrique « accuracy » adaptée à la régression, et la perte de session sur Kaggle.

**Pistes d'amélioration :** découper les données par date plutôt qu'au hasard (plus réaliste en finance), ajouter des indicateurs techniques comme les moyennes mobiles, ou tester des modèles séquentiels (LSTM).

## Exécution

```bash
pip install torch skorch scikit-learn pandas numpy matplotlib seaborn
```

Ouvrir `notebook_regression.ipynb` sur Kaggle (en ajoutant le dataset `dgawlik/nyse`) ou sur Google Colab, puis exécuter toutes les cellules.
