# NLP — BERT vs TF-IDF sur SST-2

Projet universitaire lié au **module 7 : Transformers et BERT**. L'objectif est de
comparer TF-IDF + régression logistique à BERT pour la classification binaire du
sentiment de critiques de films en anglais.

**Phase 1 validée et publiée** dans le commit
[`ecb4c82`](https://github.com/gueri-jpg/nlp-bert-vs-tfidf-sst2/commit/ecb4c82e3a36fd1c3d3d05b2efaad01e996feb5a).
**Phase 2 validée sur Google Colab, GPU Tesla T4** : `bert-base-uncased` chargé,
deux étapes d'optimisation et export des huit prédictions techniques réussis.
Le notebook exécuté dans Colab et les deux exports de phase 2 sont présents dans
le workspace. Les trois résultats TF-IDF de référence restent ceux du commit
`ecb4c82`. L'entraînement complet de BERT et le reporting comparatif attendent
une autorisation explicite de phase 3.

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
│   └── bert_vs_tfidf_sst2.ipynb   # exécuté dans Colab ; section 3 : BERT
├── results/
│   ├── .gitkeep
│   ├── tfidf_metrics.json       # généré après une exécution réussie
│   ├── tfidf_predictions.csv    # généré après une exécution réussie
│   ├── tfidf_confusion_matrix.png
│   └── phase2/
│       ├── bert_technical_report.json
│       └── bert_smoke_predictions.csv  # seulement si le smoke test réussit
├── README.md
├── requirements.txt
└── .gitignore
```

Le ZIP du cours reste local et inchangé. `.gitignore` exclut les archives, données
téléchargées, caches, environnements, checkpoints et fichiers temporaires.
Les trois résultats de la baseline peuvent être versionnés.

## Exécuter la phase 2 dans Google Colab

1. Ouvrir le [notebook de la branche main dans Colab](https://colab.research.google.com/github/gueri-jpg/nlp-bert-vs-tfidf-sst2/blob/main/notebooks/bert_vs_tfidf_sst2.ipynb).
   La section 3 validée sur GPU est incluse dans la publication de phase 2.
   Il est aussi possible d'utiliser **Fichier → Importer un notebook → Importer**
   et de choisir le notebook local exécuté.
2. Démarrer une session neuve, puis **Exécution → Modifier le type d'exécution → GPU**
   (T4 si disponible). Le CPU est possible pour le test limité avec suffisamment
   de RAM ; un GPU sera préférable pour la phase 3.
3. Exécuter **uniquement les cellules de la section 3, dans l'ordre**. Elle est
   autonome et recharge les splits officiels de SST-2. Aucun transfert manuel de
   données, poids ou résultats TF-IDF n'est nécessaire. **Ne pas utiliser « Tout
   exécuter » sur le notebook entier** : cela relancerait les exports de phase 1.
4. La première cellule de section 3 installe les dépendances. Si une bibliothèque
   déjà importée est remplacée, redémarrer la session et reprendre en section 3.
   Les API réellement présentes sont vérifiées par introspection, dont
   `TrainingArguments.eval_strategy` et `Trainer.processing_class`.
5. Lire les démonstrations WordPiece, la couverture du train et les contrôles de
   dimensions. Le code calcule la longueur avec `[CLS]` et `[SEP]`, sans troncature,
   puis choisit le plus petit seuil parmi 64, 128, 256 et 512 couvrant 99 % du train.
   Ce choix ne consulte aucun score de validation.
6. Le chargement des poids et le smoke test ne démarrent que si la mémoire libre
   atteint le seuil de précaution : 3 Gio de RAM sur CPU ; ou 2 Gio de RAM et 3 Gio
   de VRAM sur CUDA. Une insuffisance ou un dépassement mémoire est signalé dans le
   rapport ; les tests non exécutés ne sont pas déclarés réussis. Reprendre dans
   une session disposant de davantage de mémoire, en rechargeant le modèle initial.
7. Après succès, récupérer le notebook exécuté, `results/phase2/bert_technical_report.json`
   et `results/phase2/bert_smoke_predictions.csv` via le panneau **Fichiers**.
   Un rapport avec le statut `deferred_memory` signifie que le test complet du
   modèle reste à exécuter. Les fichiers Colab sont temporaires.

### Portée des contrôles BERT

Le tokenizer utilise des sous-mots WordPiece fixes ; les représentations produites
par l'encodeur BERT sont contextuelles. La section 3 montre les tokens spéciaux,
indices, masques, padding et troncature. Les textes et `idx` sont conservés dans
une vue alignée ; `labels` reprend les labels officiels sans modification. Une
vue contenant seulement les entrées du modèle et un collator assurent le padding
dynamique. Le test n'utilise jamais le split test officiel.

Le modèle chargé et validé est `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)`.
La tête de classification SST-2 est nouvellement initialisée ; l'avertissement
correspondant est attendu. L'encodeur vient du préentraînement, notamment MLM.
Tous les paramètres restent entraînables.

Le seul entraînement exécutable dans cette phase est borné à **deux mises à jour**
sur **16 exemples de train**, avec prédiction sur **8 exemples de validation**.
Le passage avant, la loss finie, les gradients et les modifications de poids dans
la tête et l'encodeur sont contrôlés. Le batch est de 2 sur CUDA ou 1 sur CPU.
Les scores du sous-échantillon sont marqués `smoke_only` et ne sont pas comparables
à la baseline. Aucun checkpoint ni poids n'est exporté dans `results/`.

Le plan de phase 3 est un dictionnaire sans lancement : 2 époques, learning rate
`2e-5`, seed 42, batch 16 sur GPU (2 sur CPU, à adapter), accuracy et F1 positif.
Après le smoke test, il faudra **recharger les poids préentraînés**, car le modèle
technique a déjà subi des mises à jour. L'entraînement sur 67 349 exemples et
l'évaluation finale sur les 872 exemples restent à autoriser.

### Conservation des prédictions individuelles

L'export réutilisable conserve `split`, `idx`, `sentence`, `true_label`,
`predicted_label`, les deux logits, `probability_positive`, `run_kind`, le nom du
modèle, la seed et `max_length`. Les labels renvoyés par Trainer doivent correspondre
à ceux du split source dans le même ordre ; le CSV est relu et vérifié.

En phase 2, seules les huit prédictions techniques peuvent être enregistrées, sous
`phase2/bert_smoke_predictions.csv`, avec `run_kind=smoke`. **Aucun fichier
`results/bert_predictions.csv` final n'est créé.** Après l'expérience complète, ce
dernier devra conserver les 872 prédictions pour permettre un appariement par
`(split, idx)`, puis une vérification des textes et labels. Les désaccords et un
éventuel test de McNemar seront étudiés ultérieurement sur ces prédictions appariées.

## Réexécuter la phase 1 dans Google Colab (historique, facultatif)

Cette procédure recalcule la baseline ; elle n'est pas nécessaire à la phase 2.

1. Ouvrir [Google Colab](https://colab.research.google.com/), puis **Fichier →
   Importer un notebook → Importer** et sélectionner
   `notebooks/bert_vs_tfidf_sst2.ipynb` depuis ce dossier local. Cette méthode
   fonctionne avant toute publication du notebook sur GitHub.
2. Choisir **Exécution → Modifier le type d'exécution → CPU**, puis
   **exécuter les cellules des sections 0 à 2 uniquement**. La première cellule
   installe les dépendances de la phase 1.
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

Les versions sont bornées dans `requirements.txt` et les cellules `%pip` de chaque
phase. La phase 2 ajoute PyTorch, Transformers 4.57, Accelerate, Safetensors et
psutil. Les versions réellement utilisées sont affichées et exportées dans les JSON. Ces bornes
ne constituent pas un verrouillage complet des dépendances : de petites variations
numériques sont possibles entre machines. Les empreintes des splits facilitent
la traçabilité sans figer une révision du Hub.

## Vérification de la phase 1

Le notebook vérifie les tailles des splits, les labels binaires, les identifiants,
la cohérence matrice/erreurs et les exports relus depuis le disque. Le CSV relu est
comparé aux textes, identifiants, labels et prédictions en mémoire ; les métriques
sont recalculées depuis ce CSV et le format PNG est contrôlé.

Exécution locale complète réussie sur **CPU, Python 3.13.14**, le 22 novembre 2026 :
les neuf cellules de code ont été exécutées dans l'ordre. Leurs sorties initiales
sont conservées dans le notebook du commit `ecb4c82` ; le notebook actuel contient
les sorties de la session Colab fournie. Apprentissage sur 67 349 exemples, évaluation sur 872,
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
La session Colab fournie a également réexécuté les sections 0 à 2 ; cette
réexécution ne remplace pas les trois résultats de référence du commit `ecb4c82`.

Les sept fichiers de phase 1 ont été publiés sur `main` dans le commit
[`ecb4c82`](https://github.com/gueri-jpg/nlp-bert-vs-tfidf-sst2/commit/ecb4c82e3a36fd1c3d3d05b2efaad01e996feb5a).
La phase 2 validée sur Tesla T4 est publiée sur `main`, avec le notebook exécuté,
la documentation, les dépendances et les deux résultats techniques légers.

## Vérification de la phase 2

**Phase 2 réussie et validée sur GPU.** Exécution Colab du **22 novembre 2026 à
18 h 30 UTC**, puis inspection des fichiers du workspace le **23 novembre 2026**.
Le rapport porte `status=passed_technical_only`, `device=cuda`, `gpu_name=Tesla T4`
et `optimizer_steps_executed=2`. Le précédent essai local avait différé le modèle
par manque de RAM ; cette limite locale n'est plus un contrôle en attente pour
la session Colab fournie.

- Environnement exécuté : Python 3.13.16, PyTorch 2.14.1+cu130, Transformers 4.57.6,
  Accelerate 1.15.0 et Hugging Face Hub 0.36.2. Les signatures d'API ont été vérifiées.
- Chargement du vrai checkpoint `bert-base-uncased` depuis `model.safetensors`,
  classe `BertForSequenceClassification`, **109 483 778 paramètres entraînables**.
  L'avertissement ne concerne que la tête binaire nouvellement initialisée
  (`classifier.weight` et `classifier.bias`), comme attendu.
- Tokenisation des 67 349 exemples de train et des 872 exemples de validation,
  alignement des `idx`, textes et labels, dimensions, masques et padding dynamique
  vérifiés. **`max_length=64` couvre 99,9941 % du train** ; quatre textes sont tronqués.
- Passage avant avec logits de dimension `(2, 2)` et loss scalaire finie, puis
  **deux mises à jour Trainer** sur le sous-ensemble de 16 exemples. Le callback
  contrôle les gradients dans la tête et l'encodeur ; les assertions confirment
  la modification de leurs poids. Pic de mémoire GPU allouée : **environ 2,06 Gio**
  pour ce test uniquement ; cela ne garantit pas la mémoire du batch de phase 3.
- **Huit prédictions exportées**, avec les `idx` dans cet ordre :
  `726, 737, 51, 482, 426, 330, 243, 7`. Le CSV porte `run_kind=smoke` et conserve
  textes, vrais labels, prédictions, logits et probabilité positive.
- Inspection sans réexécution : format du notebook valide, aucune sortie d'erreur,
  cohérence notebook/rapport, huit `idx` uniques et textes/labels identiques aux
  lignes correspondantes du CSV TF-IDF de référence. Les prédictions sont les
  argmax des logits ; les probabilités correspondent au softmax. Les métriques du
  rapport ont été recalculées depuis le CSV uniquement pour vérifier l'export.

**Ce smoke test n'est pas une évaluation finale.** Ses métriques restent dans
`smoke_only_metrics` et ne doivent entrer dans aucun tableau comparatif avec TF-IDF.

### Préservation des résultats TF-IDF de référence

Les trois fichiers locaux `results/tfidf_metrics.json`, `results/tfidf_predictions.csv`
et `results/tfidf_confusion_matrix.png` ont été comparés aux objets Git du commit
`ecb4c82` : ils correspondent à cette référence et n'ont pas été remplacés.

Le notebook Colab a réexécuté la baseline. Les empreintes `baseline_hashes_unchanged`
du rapport décrivent les fichiers régénérés dans cette session et diffèrent des
empreintes locales de référence. Le contrôle `baseline_files_unchanged=passed`
signifie qu'ils n'ont pas changé **pendant la section 3**, sans certifier leur
identité avec le commit `ecb4c82`. Conserver le rapport tel que fourni ; pour la
comparaison future, utiliser exclusivement les trois fichiers locaux de référence,
pas les sorties ou exports TF-IDF régénérés dans Colab.

### Éléments prêts pour la phase 3

Tokenizer, choix de longueur, splits alignés, padding dynamique, modèle et Trainer
sont techniquement validés sur T4. Le plan reste **2 époques, learning rate `2e-5`,
seed 42, batch train 16 et évaluation 32 sur GPU**, à ajuster si la mémoire manque.
Accuracy et F1 utilisent la classe positive 1.

Avant l'expérience complète, **recharger le checkpoint préentraîné avec la seed 42**,
sans réutiliser les poids modifiés par le smoke test. Après autorisation, entraîner
sur le train officiel et conserver les **872 prédictions finales de validation**
avec leurs `idx`, textes et labels. Elles permettront les comparaisons appariées
et, éventuellement, McNemar. Le fichier final `results/bert_predictions.csv` n'existe
pas encore et aucun entraînement complet n'a été lancé dans cette revue.

**Phase 2 validée et publication autorisée.** Attendre l'autorisation de phase 3
avant l'entraînement complet et la comparaison finale.
