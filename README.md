# NLP — BERT vs TF-IDF sur SST-2

Projet universitaire du **module 7 : Transformers et BERT**. Les phases 1 à 3
ont produit les résultats réels ; la phase 4 compare les modèles sans réentraînement.

Le rendu rédigé est [report/summary.md](report/summary.md). Le
[notebook](notebooks/bert_vs_tfidf_sst2.ipynb) conserve les expériences exécutées et
ajoute la section **« 5. Comparaison des performances BERT vs TF-IDF »**, exécutée
pour produire les tableaux, figures et le rapport final.

## Résultats vérifiés sur la validation

| Mesure | TF-IDF + LR | BERT | Écart BERT − TF-IDF |
| --- | ---: | ---: | ---: |
| Accuracy | 80,50 % | 92,55 % | +12,04 pp |
| Précision positive | 78,31 % | 92,02 % | +13,71 pp |
| Rappel positif | 85,36 % | 93,47 % | +8,11 pp |
| F1 positif | 81,68 % | 92,74 % | +11,06 pp |
| Erreurs | 170 | 65 | −105 |

Écarts calculés **avant arrondi** ; classe positive **1**. BERT réduit les erreurs
de **61,76 %**. Après appariement par `idx` : **671 succès communs**, **34 erreurs
communes**, **136 succès exclusifs de BERT**, **31 succès exclusifs de TF-IDF**.
McNemar exact bilatéral : **p ≈ 6,883 × 10⁻¹⁷**, sur 167 désaccords de correction.
Ce test concerne les taux d'erreur appariés, sans démontrer un mécanisme causal
ni la généralisation à d'autres domaines.

## Données et protocole

[SST-2 dans GLUE](https://huggingface.co/datasets/nyu-mll/glue), configuration `sst2` :
**67 349 exemples de train**, dont des fragments annotés, et **872 de validation** ;
`0 = négatif`, `1 = positif`. Aucun label du test officiel n'est utilisé.

- **TF-IDF + LR :** apprentissage uniquement sur le train ; unigrammes/bigrammes,
  `min_df=2`, au plus 100 000 caractéristiques, TF logarithmique, norme L2 ;
  régression logistique L2, `C=1`, `lbfgs`, `max_iter=1000`, seed 42.
- **BERT :** `bert-base-uncased` rechargé depuis les poids préentraînés d'origine,
  tête binaire neuve après seed 42 ; fine-tuning de l'encodeur et de la tête,
  **2 époques**, LR **`2e-5`**, `max_length=64`, padding dynamique, batch **16/32**,
  AdamW, scheduler linéaire sans warm-up, **Tesla T4 et FP16**. Évaluation du modèle
  final d'époque 2, sans sélection de la meilleure époque selon la validation.

Les JSON conservent les hyperparamètres détaillés, versions, durée et provenance.
Les deux étapes et huit prédictions du smoke test restent exclues de la comparaison.

## Fichiers du rendu

```text
notebooks/bert_vs_tfidf_sst2.ipynb   # expériences conservées et section 5 exécutée
results/
├── tfidf_metrics.json             # référence de phase 1
├── tfidf_predictions.csv
├── tfidf_confusion_matrix.png
├── phase2/                        # rapport technique et prédictions smoke
└── phase3/                        # six exports réels de BERT
    ├── bert_metrics.json
    ├── bert_predictions.csv       # 872 prédictions
    ├── bert_confusion_matrix.png
    ├── bert_training_history.csv
    ├── bert_training_curves.png
    └── bert_errors.csv            # 65 erreurs, sans catégories ajoutées
report/
├── audit_phase3_bert_tfidf.md      # audit fourni, conservé intact
├── summary.md                    # rapport scientifique final
└── figures/                      # cinq figures de comparaison et d'analyse
requirements.txt
.gitignore
```

Les résultats TF-IDF sont ceux du commit
[`ecb4c82`](https://github.com/gueri-jpg/nlp-bert-vs-tfidf-sst2/commit/ecb4c82e3a36fd1c3d3d05b2efaad01e996feb5a).
La phase 2 a été publiée dans
[`5a92b81`](https://github.com/gueri-jpg/nlp-bert-vs-tfidf-sst2/commit/5a92b813c268ad25870919ec72b0ab5a7063a08a).
Les résultats complets de phase 3 et le reporting final de phase 4 sont publiés
sur `main` avec le notebook exécuté, les résultats légers et les figures.

Les six fichiers BERT ont été décompressés depuis
`results/phase3/phase3_results.zip`, sans modifier leurs octets ni l'archive.
Le reporting accepte aussi l'archive si les fichiers décompressés sont absents ;
si les deux formes existent, leur identité est contrôlée. Poids, checkpoints,
caches et ZIP sont exclus de Git. Les résultats légers et figures sont versionnables.

## Reproduire uniquement le reporting

1. Dans l'environnement Python du projet, installer si nécessaire
   `python -m pip install -r requirements.txt jupyterlab`.
2. Lancer `python -m jupyterlab` et ouvrir le notebook local, avec les résultats
   TF-IDF et les six fichiers BERT ou leur archive présents dans l'arborescence.
3. Exécuter **uniquement les six cellules de code de la section 5**, dans l'ordre.
   Elles fonctionnent sur CPU, hors ligne, sans importer PyTorch/Transformers ni
   charger de modèle. **Ne pas utiliser « Tout exécuter »** : les sections 0 à 4
   contiennent les expériences d'apprentissage déjà terminées.
4. `report/summary.md` et les cinq PNG de `report/figures/` sont régénérés. Les
   résultats validés et l'audit préalable sont uniquement lus, puis leurs
   empreintes sont vérifiées.

Une ouverture dans Colab nécessite également les résultats avec la même
arborescence ; aucun GPU n'est nécessaire pour la section 5. Le
[lien Colab de main](https://colab.research.google.com/github/gueri-jpg/nlp-bert-vs-tfidf-sst2/blob/main/notebooks/bert_vs_tfidf_sst2.ipynb)
permet d'ouvrir le notebook publié ; les fichiers de résultats restent nécessaires
dans l'environnement d'exécution pour régénérer le reporting.

## Analyses et limites

Le rapport présente les métriques, matrices côte à côte, groupes appariés et
McNemar. Il analyse les dix erreurs réelles **205, 411, 580, 724, 770, 93, 112,
172, 183 et 728**, avec phrases, labels, deux prédictions, confiance et hypothèses
linguistiques. Ces hypothèses ne sont pas des explications causales démontrées.

La section « Analyse du risque de surapprentissage et de la confiance prédictive »
relie la baisse des fenêtres de training loss à la hausse de validation loss
(**0,253271 à 0,291790**), malgré l'amélioration simultanée de l'accuracy et du F1.
**48 des 65 erreurs finales** ont au moins 90 % de confiance, dont
**20** au-delà de 99 %. Les probabilités individuelles d'époque 1 sont absentes :
la cause de la hausse ne peut pas être isolée. Deux validations ne démontrent
pas un surapprentissage, et une seule seed ne quantifie pas la variabilité.

Les losses train **0,160300** et **0,096000** sont les dernières fenêtres de logs,
pas des moyennes par époque. La log-loss reconstruite vaut **0,291784328**, contre
**0,291790426** enregistrée par Trainer : écart numérique d'environ **6,10 × 10⁻⁶**,
signalé sans modifier les sources. Aucun écart sur les métriques de classification,
matrices ou effectifs appariés n'a été trouvé.

## Vérifications de phase 4

Recalcul depuis les CSV, contrôle des JSON et du CSV des 65 erreurs ; appariement
strict des 872 identifiants, textes et labels ; contrôle des logits, argmax,
softmax et indicateurs `correct` ; vérification de l'historique et du test exact
par somme binomiale indépendante. Les six nouvelles cellules sont exécutées,
les figures et liens locaux contrôlés, les sources et l'audit conservés.

Les cellules et sorties des expériences restent intactes ; seule la mention
Markdown obsolète de section 4 indiquant l'absence d'entraînement a été corrigée.
Le rapport inclut une conclusion de sept lignes reliant TF-IDF, MLM,
représentations contextuelles, `[CLS]` et fine-tuning supervisé.

**Phase 4 terminée, publication autorisée. Les cellules de reporting ne lancent
aucun réentraînement.**
