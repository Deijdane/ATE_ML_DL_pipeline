# Pipeline de post-processing pour TermSuite

Ce projet fournit une pipeline d’outils Machine/Deep Learning permettant de classifier et réduire le bruit des résultats de [TermSuite](http://termsuite.github.io/) et d’extraire les contextes définitoires liés aux termes extraits.

---

## ⚠️ Prérequis

- **Modèle BERT** : le fichier `base_de_bert.pt` n’est pas présent dans le dépôt (limite GitBucket).  
  Téléchargez-le ici : https://huggingface.co/feyhre/bert_ADE 
  Placez-le ensuite dans le même dossier que les scripts de la pipeline.

- **Bibliothèques Python** :  
  ```bash
  pip install -r requirements.txt
  ```

- **Dossier results** : créez un sous-dossier results/ (utilisé pour stocker les sorties intermédiaires).

## Données attendues

Le modèle de classification repose sur des features complémentaires.
Lors de l’exécution de TermSuite, demandez les propriétés suivantes :

```
--tsv-properties "documentFrequency,specificity,frequency,IndependantFrequency,Independance,tf-idf,SwtSize"
```

Votre fichier TSV doit contenir a minima les colonnes suivantes :
"key, dFreq, spec, freq, iFreq, ind, tfIdf, swtSize"

## Pipeline complète :

### 0. pipeline_full.sh
```
./pipeline_full.sh $PATH_INPUT $PATH_CORPUS $PATH_OUTPUT \
    --RELU_PATH ... \
    --TS_SEUIL_FREQ ... \
    --TS_BATCH_SIZE ... \
    --TS_VAL_THRESH ... \
    --TS_KEEP_NOISE ... \
    --BERT_PATH ... \
    --DE_BATCH_SIZE ... \
    --DE_VAL_THRESH ...
```

- PATH_INPUT : extraction TermSuite (.tsv)
- PATH_CORPUS : dossier contenant les fichiers .txt du corpus
- PATH_OUTPUT : chemin vers le résultat final (HTML)

Options principales :

- --RELU_PATH : chemin vers le modèle ReLU (classification TermSuite)
- --TS_SEUIL_FREQ : seuil de fréquence (defaut: 5, 0 = conserver tout)
- --TS_BATCH_SIZE : taille des batchs (defaut: 32)
- --TS_VAL_THRESH : seuil de confiance du modèle (defaut: 0.143)
- --TS_KEEP_NOISE : conserver le bruit (True/False)
- --BERT_PATH : chemin vers le modèle BERT (classification contextes)
- --DE_BATCH_SIZE : taille des batchs (defaut: 32)
- --DE_VAL_THRESH : seuil de confiance du modèle (defaut: 0.5)

## Étapes de la pipeline

La pipeline comporte 4 grandes étapes.

### 1. ts_classif.py

Classifie les termes extraits par TermSuite (concept vs bruit). Retourne un fichier .csv regroupant une liste de termes validé par le modèle avec leur label de validation et une probabilité représentant la confiance du modèle.

```
python ts_classif.py $PATH_INPUT $PATH_OUTPUT \
    --model_path $RELU_PATH \
    --seuil_freq $TS_SEUIL_FREQ \
    --batch_size $TS_BATCH_SIZE \
    --threshold $TS_VAL_THRESH \
    --keep_noise $TS_KEEP_NOISE
```

### 2. de_prestep.py

Récupère, via regex, les phrases contenant les termes. Retourne un fichier .csv avec une liste de phrase, les termes validé présent dans ces phrases et le document source.


```
python de_prestep.py $PATH_INPUT $PATH_CORPUS $PATH_OUTPUT
```


### 3. de_classif.py

Classifie les phrases issues de de_prestep.py (contexte définitoire ou non).
⚠️ La classification se fait indépendamment du terme associé.

```
python de_classif.py $PATH_INPUT $PATH_OUTPUT \
    --model_path $BERT_PATH \
    --batch_size $DE_BATCH_SIZE \
    --threshold $DE_VAL_THRESH
```

### 4. de_csv_html.py

Convertit les résultats de de_classif.py en fichier HTML lisible.

```
python de_csv_html.py $PATH_INPUT $PATH_OUTPUT
```

## Résumé

- Préparer les données TermSuite (TSV avec features complémentaires)

- Créer results/ et télécharger base_de_bert.pt

- Lancer la pipeline complète avec pipeline_full.sh

- Obtenir un HTML final contenant les contextes définitoires.
