# Audit de la phase 3 — Comparaison BERT vs TF-IDF sur SST-2

> Analyse reproduite à partir de `phase3_results.zip` et du fichier `tfidf_predictions.csv` de la phase 1. Les interprétations des erreurs sont des hypothèses, pas des explications causales démontrées.

## Protocole

- Classification binaire SST-2 : 67 349 phrases train, 872 validation, mêmes identifiants et labels.
- TF-IDF (unigrammes/bigrammes) + régression logistique, seed 42.
- BERT `bert-base-uncased`, fine-tuning 2 époques sur Tesla T4, LR 2e-5, seed 42, max length 64, batch train 16, eval 32, FP16.
- Modèle final de la deuxième époque évalué sur les 872 exemples de validation.

## Métriques recalculées indépendamment

| Mesure | TF-IDF + LR | BERT |
|---|---:|---:|
| Accuracy | 0.805046 | 0.925459 |
| Précision positive | 0.783058 | 0.920177 |
| Rappel positif | 0.853604 | 0.934685 |
| F1 positif | 0.816810 | 0.927374 |
| Correctes | 702 | 807 |
| Erreurs | 170 | 65 |

- Gain d’accuracy : **12.04 points**.
- Réduction des erreurs : **61.76 %**.
- TF-IDF matrice : `[[323, 105], [65, 379]]` (lignes = réels, colonnes = prédits).
- BERT matrice : `[[392, 36], [29, 415]]`.

## Tableau apparié et test de McNemar

| Situation | Exemples |
|---|---:|
| Les deux corrects | 671 |
| BERT correct, TF-IDF incorrect | 136 |
| TF-IDF correct, BERT incorrect | 31 |
| Les deux incorrects | 34 |

Test exact de McNemar bilatéral, sur les 167 désaccords de correction : **p = 6.883e-17**. Il indique une différence statistiquement marquée des taux d’erreur sur cette validation ; il ne prouve ni l’origine causale du gain ni une généralisation universelle.

## Évolution de BERT

| Époque | Training loss (journal) | Validation loss | Accuracy | F1 positif |
|---|---:|---:|---:|---:|
| 1 | 0.160300 | 0.253271 | 0.918578 | 0.920670 |
| 2 | 0.096000 | 0.291790 | 0.925459 | 0.927374 |

- Le modèle se trompe avec au moins 90 % de confiance dans **48 des 65 erreurs** ; avec plus de 99 % dans 20 cas. Cela peut contribuer à la hausse de la log-loss sans en constituer une explication causale établie.
- Ne pas appeler automatiquement « surapprentissage démontré » une validation loss en hausse sur deux époques ; parler plutôt de signal à surveiller.

## Dix erreurs de BERT à examiner qualitativement

Valeurs 0 = négatif, 1 = positif. La colonne TF-IDF indique la prédiction de la baseline. **Catégorisation proposée à examiner manuellement**.

| `idx` | Phrase originale SST-2 | Réel | BERT | TF-IDF | Catégorie hypothétique | Interprétation prudente |
|---:|---|---:|---:|---:|---|---|
| 205 | falls neatly into the category of good stupid fun . | 1 | 0 | 1 | Vocabulaire positif et péjoratif mêlé | Le terme « stupid » est négatif isolément, mais « good stupid fun » peut exprimer un plaisir assumé. |
| 411 | i do n't mind having my heartstrings pulled , but do n't treat me like a fool . | 0 | 1 | 0 | Négation et opposition | La concession initiale est suivie d’un reproche explicite ; le modèle semble retenir un jugement positif. |
| 580 | it takes a certain kind of horror movie to qualify as ` worse than expected , ' but ghost ship somehow manages to do exactly that . | 0 | 1 | 0 | Comparaison défavorable | « worse than expected » exprime une aggravation négative malgré une construction idiomatique. |
| 724 | it 's everything you do n't go to the movies for . | 0 | 1 | 0 | Négation idiomatique | L’expression « everything you do not go to the movies for » signifie un rejet global. |
| 770 | i do n't think i laughed out loud once . | 0 | 1 | 0 | Négation d’une émotion positive | « don’t think I laughed ... once » signale l’absence de satisfaction. |
| 93 | if steven soderbergh 's ` solaris ' is a failure it is a glorious failure . | 1 | 0 | 0 | Concession / sens global | « glorious failure » combine un échec explicite et une valorisation paradoxale. |
| 112 | hilariously inept and ridiculous . | 1 | 0 | 0 | Humour / ambiguïté lexicale | « hilariously inept » peut être apprécié comme comique malgré les adjectifs péjoratifs. |
| 172 | it seems like i have been waiting my whole life for this movie and now i ca n't wait for the sequel . | 1 | 0 | 0 | Négation à portée positive | « can’t wait for the sequel » est idiomatiquement enthousiaste. |
| 183 | the lower your expectations , the more you 'll enjoy it . | 0 | 1 | 1 | Éloge conditionnel / ironie possible | L’appréciation conditionnée à de faibles attentes peut être globalement négative. |
| 728 | but taken as a stylish and energetic one-shot , the queen of the damned can not be said to suck . | 1 | 0 | 0 | Négation d’un jugement négatif | « cannot be said to suck » atténue ou inverse la critique défavorable. |

## Précautions méthodologiques

- Les catégories qualitatives ne démontrent pas le mécanisme interne de BERT ; elles décrivent des caractéristiques visibles du texte.
- Certaines étiquettes de SST-2 peuvent surprendre ; vérifier le sens global avant de qualifier une erreur de « négation » ou « ironie ».
- Les métriques proviennent d’un seul run et d’un seul split de validation : ne pas extrapoler de façon excessive.
- Le test de McNemar est optionnel dans le livrable académique mais correctement applicable à ces prédictions appariées.
- Conserver le checkpoint hors GitHub ; pousser les fichiers de résultats légers, le notebook et la documentation si souhaité.
- Corriger le texte Markdown de la section 4 du notebook indiquant encore « aucune expérience complète exécutée » : ce texte décrit le Temps A mais les sorties correspondent maintenant à une exécution complète.

## Suite recommandée

Phase 4 : intégrer le tableau comparatif, les deux matrices de confusion, les 10 erreurs commentées et une conclusion de 5–10 lignes reliant MLM, [CLS], représentations contextuelles et fine-tuning ; expliquer également la hausse de validation loss malgré le gain d’accuracy.
