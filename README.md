# Lab 1, Partie 1 : Régression avec un DNN/MLP en PyTorch (dataset NYSE)

**Objectif :** prendre en main PyTorch pour résoudre une tâche de régression avec un réseau de neurones de type MLP.
**Problème :** prédire le prix de clôture (`close`) d'une action à partir de `open`, `low`, `high` et `volume`.

**Dataset :** [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse), fichier `prices-split-adjusted.csv` (les prix y sont corrigés des splits d'actions) : 851 264 lignes, 501 actions, du 04/01/2010 au 30/12/2016.
**Outils :** Kaggle (GPU), PyTorch, scikit-learn, skorch, pandas, matplotlib, seaborn.
Les graphes et les sorties détaillées sont dans le notebook `lab1-tp.ipynb`.

---

## 1. Analyse exploratoire (EDA)

- Aucune valeur manquante, aucun doublon.
- Prix moyen d'environ 65 $, médiane d'environ 48 $, maximum d'environ 1 600 $.
- Les quatre variables de prix (`open`, `high`, `low`, `close`) ont des distributions quasi identiques et très asymétriques à droite : la plupart des valeurs sont entre 0 et 100 $, avec une longue queue.
- Le `volume` est encore plus écrasé près de zéro (maximum d'environ 8,6 × 10⁸). Ses échelles très différentes de celles des prix rendent la **normalisation indispensable**.
- `open`, `high`, `low` et `close` sont corrélés à environ 0,999, ce qui est logique : le prix de clôture d'une journée reste proche de son prix d'ouverture. La tâche est donc plutôt facile.
- Les boxplots montrent beaucoup de valeurs extrêmes, surtout pour le volume. Je les ai conservées : ce sont des actions très chères ou des jours d'activité exceptionnelle, donc des situations réelles du marché.

## 2. Architecture du réseau

- Découpage 80 % / 20 % : 681 011 lignes en train, 170 253 en test.
- `StandardScaler` ajusté **uniquement sur le train** (variables et cible) pour éviter la fuite de données.
- MLP : 4 entrées, couches cachées `Linear + ReLU`, 1 neurone de sortie **sans activation** (valeur continue).
- Perte : **MSE** (maximum de vraisemblance sous bruit gaussien). ReLU est choisie pour éviter le gradient qui disparaît.
- Boucle d'entraînement manuelle : `zero_grad`, forward, `backward`, `step`. Les batchs sont pris directement dans les tenseurs, car le `DataLoader` était trop lent.
- L'accuracy de classification n'existe pas en régression : j'utilise le **R²** (1 = parfait).

## 3. GridSearch

`GridSearchCV` avec `skorch`, validation croisée à 3 plis, sur un **sous-échantillon de 50 000 lignes** (batch de 256) pour limiter le temps de calcul. 36 combinaisons, soit 108 entraînements.

| Hyperparamètre | Valeurs testées |
|---|---|
| Learning rate | 0.01, 0.001 |
| Optimiseur | Adam, SGD, RMSprop |
| Epochs | 20, 40 |
| Architecture | (32), (64, 32), (128, 64, 32) |

**Meilleure configuration :** lr = 0.001, optimiseur = Adam, epochs = 40, architecture = (32) (une couche cachée de 32 neurones), **R² CV = 0,99987**.

Les meilleures configurations utilisent presque toutes Adam avec lr = 0.001. Les trois architectures donnent des scores très proches (0,99987 pour (32), 0,99985 pour (64, 32), 0,99981 pour (128, 64, 32)) : le problème étant quasi linéaire, un réseau plus profond n'apporte rien. SGD apparaît dans le top 10 mais reste derrière Adam.

## 4. Courbes Loss/Epochs et Accuracy (R²)/Epochs

Le modèle 1 est entraîné sur tout le train avec la meilleure configuration (batch de 1024).

| | Loss (MSE) | R² |
|---|---|---|
| Train | 0,00006 | 0,99994 |
| Test | 0,00007 | 0,99993 |

La loss chute rapidement durant les premières epochs (de 0,00053 à environ 0,0001) puis se stabilise autour de 10⁻⁴ avec quelques oscillations. Le R² dépasse 0,9999 en train comme en test. Les courbes train et test sont quasiment confondues : **ni overfitting, ni underfitting**. Les petites oscillations en fin d'entraînement indiquent un learning rate un peu élevé pour la phase finale.

## 5. Régularisation

J'ai comparé le modèle de base à cinq variantes sur la même architecture et la même initialisation : L2 (weight decay 1e-4), Dropout (0,2), BatchNorm, Early Stopping (patience 5) et une combinaison L2 + Dropout + BatchNorm.

| Modèle | Train loss | Test loss | Train R² | Test R² | Écart R² | Epochs |
|---|---|---|---|---|---|---|
| 1-Base | 0.00006 | 0.00007 | 0.99994 | 0.99993 | 0.00000 | 40 |
| 2-L2 | 0.00009 | 0.00009 | 0.99991 | 0.99991 | 0.00000 | 40 |
| 3-Dropout | 0.00088 | 0.00089 | 0.99912 | 0.99911 | 0.00000 | 40 |
| 4-BatchNorm | 0.00804 | 0.00816 | 0.99196 | 0.99189 | 0.00007 | 40 |
| 5-EarlyStop | 0.00010 | 0.00010 | 0.99990 | 0.99990 | 0.00000 | 16 |
| 6-Combiné | 0.00656 | 0.00663 | 0.99344 | 0.99341 | 0.00002 | 40 |

**Analyse :**
- Le modèle de base reste le meilleur. Comme l'écart train/test est déjà nul, il n'y a **pas d'overfitting à corriger** : la régularisation n'a rien à améliorer.
- Dropout, BatchNorm et la combinaison **dégradent** les résultats et rendent les courbes instables (le R² de la combinaison chute jusqu'à environ 0,95 au cours de l'entraînement). Ces techniques ajoutent du bruit à un modèle qui n'en avait pas besoin, sur un problème quasi linéaire avec une cible continue.
- L2 donne presque le même résultat que la base (0,99991).
- L'**Early Stopping** est la plus intéressante : R² de 0,9999 en 16 epochs au lieu de 40.

**Limite :** le modèle 1 a été entraîné avec un batch de 1024, alors que les variantes utilisent un batch de 256. La comparaison n'est donc pas strictement identique, et une partie de l'écart entre la base et les variantes pourrait venir de ce changement de batch. Les conclusions (régularisation inutile sans overfitting) restent valables, puisque la base fait déjà R² = 0,9999.

## 6. Synthèse personnelle

Dans ce TP, j'ai découvert comment construire un réseau de neurones avec PyTorch, de la préparation des données jusqu'à l'évaluation. J'ai compris le rôle de chaque étape de la boucle d'entraînement (forward, calcul de la perte, backpropagation, mise à jour des poids) et comment ReLU donne au réseau sa capacité non linéaire.

J'ai aussi vu l'importance des hyperparamètres : le learning rate et l'optimiseur influencent la vitesse de convergence, et le GridSearch permet de les choisir de façon systématique, même s'il coûte cher en temps de calcul (d'où le sous-échantillon de 50 000 lignes).

Sur la régularisation, j'ai compris qu'elle sert à lutter contre l'overfitting mais qu'elle n'améliore pas les résultats quand le modèle ne sur-apprend pas, comme ici. Elle peut même les dégrader.

**Difficultés :** le temps de calcul (le `DataLoader` était lent, je l'ai remplacé par un découpage direct des tenseurs), le choix d'une métrique d'« accuracy » adaptée à la régression, la fuite de données lors de la normalisation et la perte de session sur Kaggle.
**Pistes d'amélioration :** découper les données par date plutôt qu'au hasard (plus réaliste en finance), ajouter des indicateurs techniques (moyennes mobiles), utiliser le même batch pour toutes les variantes, ou tester des modèles séquentiels (LSTM).

## Exécution

```bash
pip install torch skorch scikit-learn pandas numpy matplotlib seaborn
```

Ouvrir le notebook sur Kaggle (en ajoutant le dataset `dgawlik/nyse`) ou sur Google Colab, puis exécuter toutes les cellules.
