# NLP — BERT vs TF-IDF sur SST-2

Projet universitaire lié au **module 7 : Transformers et BERT**. L'objectif est de
comparer TF-IDF + régression logistique à BERT pour la classification binaire du
sentiment de critiques de films en anglais.

**Phase 1 uniquement :** exploration de SST-2 et baseline scikit-learn sur CPU.
BERT, son tokenizer, le fine-tuning et le reporting comparatif seront développés
dans les phases suivantes, après validation explicite.

## Données et protocole

[SST-2 dans GLUE](https://huggingface.co/datasets/nyu-mll/glue) est chargé avec
`load_dataset("nyu-mll/glue", "sst2")` : **67 349** exemples de train et **872** de
validation ; `0 = négatif`, `1 = positif`. Aucun label du test n'est utilisé.

Un `Pipeline` apprend le vocabulaire, les IDF et la régression logistique uniquement
sur le train officiel. Hyperparamètres fixés avant évaluation : unigrammes et
bigrammes, `min_df=2`, au plus 100 000 caractéristiques, TF logarithmique,
normalisation L2 ; régression logistique L2, `C=1`, `lbfgs`, `max_iter=1000`, seed 42.
Aucune sélection de modèle sur la validation. Accuracy, précision, rappel et F1
sont mesurés sur la validation ; la classe positive des métriques binaires est 1.

Le notebook présente les effectifs, exemples annotés, distributions de sentiments,
longueurs en mots/caractères, graphiques, matrice de confusion et jusqu'à dix
erreurs. Un contrôle signale les textes normalisés identiques entre les splits.
Les splits officiels restent intacts ; ce contrôle ne détecte pas toutes les
paraphrases ni tous les fragments communs. La validation ne constitue pas un test
indépendant si elle guide les choix des phases suivantes.

## Structure

```text
nlp-bert-vs-tfidf-sst2/
├── notebooks/
│   └── bert_vs_tfidf_sst2.ipynb   # livrable principal, sections 0 à 2
├── results/
│   ├── .gitkeep
│   ├── tfidf_metrics.json       # généré après une exécution réussie
│   ├── tfidf_predictions.csv    # généré après une exécution réussie
│   └── tfidf_confusion_matrix.png
├── README.md
├── requirements.txt
└── .gitignore
```

Le ZIP du cours reste local et inchangé. `.gitignore` exclut les archives, données
téléchargées, caches, environnements, checkpoints et fichiers temporaires.
Les trois résultats de la baseline peuvent être versionnés.

## Exécuter dans Google Colab

1. Ouvrir [Google Colab](https://colab.research.google.com/), puis **Fichier →
   Importer un notebook → Importer** et sélectionner
   `notebooks/bert_vs_tfidf_sst2.ipynb` depuis ce dossier local. Cette méthode
   fonctionne avant toute publication du notebook sur GitHub.
2. Choisir **Exécution → Modifier le type d'exécution → CPU**, puis **Exécution →
   Tout exécuter**. La première cellule installe les dépendances de la phase 1.
3. Si Colab demande un redémarrage après installation, redémarrer et relancer tout.
   Une connexion Internet est nécessaire pour les dépendances et SST-2 ; aucun
   GPU, compte Hugging Face ou secret n'est nécessaire.
4. Vérifier l'absence d'erreur, les tailles des splits et les métriques affichées.
5. Récupérer `results/phase1_results.zip`, généré par la dernière cellule. Si le
   téléchargement automatique est bloqué, utiliser le panneau **Fichiers → results**.
   L'archive contient les trois fichiers de résultats ci-dessous.
6. Télécharger aussi le notebook **avec ses sorties** via **Fichier → Télécharger
   → Télécharger le fichier .ipynb**. Conserver les résultats avant de fermer la
   session : les fichiers de la session Colab sont temporaires.

À fournir pour validation :

- Le notebook exécuté, avec les graphiques et les exemples d'erreurs.
- `tfidf_metrics.json` : quatre métriques entre 0 et 1, nombre d'erreurs,
  matrice, hyperparamètres, versions et empreintes des splits.
- `tfidf_predictions.csv` : 872 lignes, colonnes `idx`, `sentence`, `true_label`,
  `predicted_label`, dans l'ordre officiel de la validation.
- `tfidf_confusion_matrix.png` : lignes = sentiment réel, colonnes = sentiment
  prédit ; ordre des classes `[0, 1]`.

## Exécution locale et compatibilité

Python **3.11 à 3.13** est recommandé. Dans un environnement virtuel, installer
`python -m pip install -r requirements.txt jupyterlab`, puis lancer
`python -m jupyterlab` et exécuter le notebook dans l'ordre.

Les chemins utilisent `pathlib` : les sorties vont dans `results/` à la racine du
projet depuis `notebooks/`, et dans `./results/` dans Colab. Les calculs utilisent
des matrices creuses sur CPU. Une erreur de connexion à Hugging Face nécessite de
rétablir l'accès réseau puis de relancer la cellule de chargement ; elle ne doit
jamais être remplacée par des résultats inventés. Un avertissement de convergence
du classifieur arrête l'entraînement et doit être examiné.

Les versions sont bornées dans `requirements.txt` et dans la cellule `%pip` ; les
versions réellement utilisées sont affichées et exportées dans le JSON. Ces bornes
ne constituent pas un verrouillage complet des dépendances : de petites variations
numériques sont possibles entre machines. Les empreintes des splits facilitent
la traçabilité sans figer une révision du Hub.

## Vérification de la phase 1

Le notebook vérifie les tailles des splits, les labels binaires, les identifiants,
la cohérence matrice/erreurs et les exports relus depuis le disque. Le CSV relu est
comparé aux textes, identifiants, labels et prédictions en mémoire ; les métriques
sont recalculées depuis ce CSV et le format PNG est contrôlé.

Exécution locale complète réussie sur **CPU, Python 3.13.14**, le 22 novembre 2026 :
les neuf cellules de code ont été exécutées dans l'ordre et leurs sorties sont
conservées dans le notebook. Apprentissage sur 67 349 exemples, évaluation sur 872,
convergence en 49 itérations et vocabulaire de 74 055 caractéristiques.

| Mesure de validation | Résultat réel |
| --- | ---: |
| Accuracy | 0,805046 |
| Précision (classe 1) | 0,783058 |
| Rappel (classe 1) | 0,853604 |
| F1 (classe 1) | 0,816810 |
| Erreurs | 170 / 872 |

Matrice de confusion : `[[323, 105], [65, 379]]`. Aucun texte entier commun aux
splits n'a été détecté après normalisation de la casse et des espaces.

Vérifications réellement effectuées :

- Validation du format Jupyter avec `nbformat` et analyse de syntaxe de toutes les
  cellules, y compris la commande `%pip` après transformation IPython.
- Exécution complète dans un noyau Jupyter local avec `nbclient`, sans erreur.
- Vérification du seul appel d'ajustement : `baseline.fit(X_train, y_train)` ;
  absence d'import BERT et de lecture du split test.
- Tests isolés des métriques avec un cas asymétrique vérifié à la main, absence
  de prédictions positives et rejet de cinq cas d'entrées invalides.
- Test du pipeline avec un terme propre à la validation : ce terme n'entre pas
  dans le vocabulaire ; vocabulaire et IDF restent identiques après prédiction.
- Relecture des 872 prédictions, recalcul des métriques et de la matrice, décodage
  du PNG et inspection visuelle de sa lisibilité.
- Contrôle `pip check` réussi, vérification des exclusions Git et comparaison
  SHA-256 du ZIP du cours avant/après l'exécution : archive inchangée.

Un premier essai a révélé une dépendance facultative de `DataFrame.style` ;
l'affichage utilise désormais `DataFrame.round`, puis le notebook a été réexécuté
intégralement avec succès. Les outils de vérification locaux (`nbformat`,
`nbclient`, `ipykernel`) sont distincts des dépendances nécessaires dans Colab.
L'exécution dans l'interface Google Colab et son téléchargement automatique
restent à vérifier par l'utilisateur.

Le dossier a été initialisé sur la branche `main`, avec `origin` pointant vers
`https://github.com/gueri-jpg/nlp-bert-vs-tfidf-sst2.git`. Aucun commit ni push n'a
été réalisé : les fichiers sont locaux et ne sont pas encore publiés sur GitHub.

**Arrêt après cette phase.** Attendre : « Phase 1 validée, passe à la phase 2. »
Aucun commit ni push automatique.
