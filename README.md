Pipeline de post-processing pour TermSuite
===============

Pipeline d'outils Machine/Deep Learning permetant de classifier et réduire le bruit des résultats de Termsuite et d'extraire les contexte définitoire lié aux termes extraits.

==ATTENTION==

Due à une limite de taille de fichier sur le gitbucket, le modèle "base\_de\_bert.pt" ne peut pas être uploadé. Vous pouvez le trouver en téléchargement ici : https://huggingface.co/feyhre/bert_ADE
Une fois téléchargé, placer le dans le dossier avec le reste des fichiers constituant la pipeline

Assurez-vous d'installez les bibliothèques requises (```pip install -r requirements.txt```)

Créer un dossier results dans le même sous-dossier. Il est utilisé pour stocker les résultats durant l'utilisation de la pipeline complète.


===============

Cette Pipeline Post-precess les résultats de TermSuite, avec pour but principal de réduire le bruit, puit de récupérer les context définitoire lié à ces termes directement dans le même corpus.

Le modèle de classification pour TermSuite se base sur des features compémentaires. Pour assurer un bon fonctionnement de la pipeline, assurez vous de demander les features complémentaires suivantes lors de l'exécution de Termsuite : 

 --tsv-properties "documentFrequency,specificity,frequency,IndependantFrequency,Independance,tf-idf,SwtSize"

Votre TSV devrait donc avoir a minima les colones suivantes : 
"key", "dFreq", "spec", "freq", "iFreq", "ind", "tfIdf", "swtSize"


La pipeline a 4 étapes, détaillé plus bas.

### pipeline_full.sh

la pipeline à besoin d'un dossier "results"  dans le même sous_dossier pour stocker les outputs généré durant la production.

```
./pipeline_full.sh $PATH_INPUT PATH_CORPUS PATH_OUTPUT --RELU_PATH --TS_SEUIL_FREQ --TS_BATCH_SIZE --TS_VAL_THRESH --TS_KEEP_NOISE --BERT_PATH --DE_BATCH_SIZE --DE_VAL_THRESH
```

détail :

- PATH_IMPUT : path de l'extration TermSuite.
- PATH_CORPUS : path du dossier contenant les fichier .txt servant au corpus.
- PATH_OUTPUT : path pour l'output final (HTML uniquement).

- --RELU_PATH : path du modèle ReLU pour la classification TermSuite.
- --TS_SEUIL_FREQ : Seuil de fréquence pour les termes issues de TermSuite. base = 5. Pour conserver tout les termes, utiliser 0.
- --TS_BATCH_SIZE : Size des batch pour classification TermSuite. base = 32.
- --TS_VAL_THRESH : Threshold de confiance du modèle. base = 0.143.
- --TS_KEEP_NOISE : Booléen (True or False). Défini si le noise est conservé ou jeté.

- --BERT_PATH : path du modèle BERT pour la classification des contextes définitoires.
- --DE_BATCH_SIZE : Size des batch pour classification des contextes définitoires. base = 32.
- --DE_VAL_THRESH : Threshold de confiance du modèle. base = 0.5

### ts_classif.py

Classifieur Linear/ReLU à appliquer directement sur les résultats de TermSuite. Le Classifieur calcule une probabilité que chaque terme soit un potentiel concept ou du bruit.

pour l'utiliser :

```
ts_classif.py $PATH_INPUT $PATH_OUTPUT --model_path $RELU_PATH --seuil_freq $TS_SEUIL_FREQ --batch_size $TS_BATCH_SIZE --threshold $TS_VAL_THRESH --keep_noise $TS_KEEP_NOISE 
```

détail :

- PATH_IMPUT : path de l'extration TermSuite.
- PATH_OUTPUT : path pour l'output.

- --RELU_PATH : path du modèle ReLU pour la classification TermSuite.
- --TS_SEUIL_FREQ : Seuil de fréquence pour les termes issues de TermSuite. base = 5. Pour conserver tout les termes, utiliser 0.
- --TS_BATCH_SIZE : Size des batch pour classification TermSuite. base = 32.
- --TS_VAL_THRESH : Threshold de confiance du modèle. base = 0.143.
- --TS_KEEP_NOISE : Booléen (True or False). Défini si le noise est conservé ou jeté.

### de_prestep.py

Utilise regex pour collecter les phrases contenant les termes. Prend en input un fichier .csv séparé par ";" contenant au moins une colonne "term". Output un .csv contenant des phrases, les termes trouvés dans ladite phrase (minimum 1, listé), et le document source dont vient la phrase.

```
python de_prestep.py $PATH_IMPUT $PATH_CORPUS $PATH_OUTPUT
```

détail :

- PATH_IMPUT : Path du fichier d'input .csv.
- PATH_CORPUS : path du dossier contenant les fichier .txt servant au corpus. Recomande d'utiliser les mêmes que pour TermSuite.
- PATH_OUTPUT : path pour l'output.

### de_classif.py

Classifieur Linear/ReLU à appliquer sur les résultats de de_prestep.py Le Classifieur calcule une probabilité que chaque phrase soit un contexte définitoire.
Attention, le classifieur ne prend pas en compte les termes présent. Il peut détecter un contexte définitoire qui pourtant appartiendrait à un terme différent de celui associé par de_prestep.py.
 
```
python de_classif.py $PATH_INPUT $PATH_OUTPUT --model_path $BERT_PATH --batch_size $DE_BATCH_SIZE --threshold $DE_VAL_THRESH
```

- PATH_IMPUT : path du .csv issue de de_prestep.py.
- PATH_OUTPUT : path pour l'output final (csv).

- --BERT_PATH : path du modèle BERT pour la classification des phrases.
- --DE_BATCH_SIZE : Size des batch pour classification des phrases. base = 32.
- --DE_VAL_THRESH : Threshold de confiance du modèle. base = 0.5.

### de_csv_html.py

Petit programme simple prennant les résultats de de_classif.py et les transforme en un fichier html lisible

```
python de_csv_html.py $PATH_INPUT $PATH_OUTPUT
```

- PATH_IMPUT : path du .csv issue de de_classif.py.
- PATH_OUTPUT : path pour l'output final (HTML).
