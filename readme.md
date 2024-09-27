
# Benchmarking on MIMIC-III Dataset

## Reference

Sanjay Purushotham*, Chuizheng Meng*, Zhengping Che, and Yan Liu. "[Benchmarking Deep Learning Models on Large Healthcare Datasets.](https://www.sciencedirect.com/science/article/pii/S1532046418300716)" Journal of Biomedical Informatics (JBI). 2018.

An earlier version is available on arXiv ([arXiv preprint arXiv:1710.08531](https://arxiv.org/abs/1710.08531)).

## Requirements

### Database

You must have the database [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) running on your local machine or on a server. To construct the database locally, see [the official guidance](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/buildmimic/postgres/README.md).

You have to fill necessary credentials to connect to the database:
```bash
cp preprocessing/config/connection_template.json preprocessing/config/connection.json
# Fill all keys listed under "mimiciii" in connection.json.
```

### Packages

- For data preparation and SuperLearner:
    - Run `bash install.sh`.
- For Feedforward Network and Feedforward Network+LSTM:
    - Anaconda 2==4.4.0
    - Theano==0.9.0
    - Keras==2.0.6

## Prepare data for benchmarking
### Generate input files

```bash
python -m preprocessing.preprocess --cachedir data --num_workers <number of processes>
```

The preprocessing should finish within 1 day with `--num_workers 4`.

All input files are stored under `data/` and require 36GB disk space.

We also provide processed data hosted on [Google Drive](https://drive.google.com/file/d/1URr0d6ZL4jBygM5trel260SoisDmJ536/view?usp=drive_link) (only containing necessary files for training following models). Please contact Chuizheng Meng (mengcz95thu@gmail.com) if you have any questions.

### Evaluate performance

<!-- #### SuperLearner(R version)

Use the following command to run [SuperLearner(R version)](https://cran.r-project.org/package=SuperLearner) on specific dataset:

```bash
Rscript [WD]/r_process_mimic_all.r [path to the folder storing input_F_T.csv and output_F_T.csv, no slash in the end] [rank of label]
```

For example, run SuperLearner-I(which means using subscores as features) with 17 processed features from MIMIC-III full dataset on in-hospital mortality prediction task:

```bash
Rscript [WD]/r_process_mimic_all.r [DD]/admdata_17f/24hrs/non_series/folds 0
```

The result is saved with the name `results_[rank of label].rds` under the same folder with where input files and output files are located. You can use `13_r_validation.ipynb` to parse the result and calculate the metrics.

SuperLearner(R version) only supports mortality prediction tasks since its efficiency is too low and we do not plan to apply it to other tasks. -->

#### SuperLearner(Python version)

By default the path to the main program of [SuperLearner(Python Version)](https://github.com/lendle/SuPyLearner) is `[RD]/Codes/SuperLearnerPyVer/python/superlearner_pyver.py`.

Use the following command to run SuperLearner(Python version):

```bash
python [path to the main program('superlearner_pyver.py') of SuperLearner(Python version)] [path to 'non_series' folder of a dataset] [path to 'series' folder of a dataset] [task name, 'mor'/'los'/'icd9'] [rank of label] [name of subset, 'all'/'cv'/'mv'] [method name, 'sl1'/'sl2']
```

For example, run SuperLearner-II with 17 raw features from MIMIC-III full dataset on length of stay prediction task:

```bash
python superlearner_pyver.py ../../../Data/admdata_17f/24hrs_raw/non_series ../../../Data/admdata_17f/24hrs_raw/series los 0 all sl2
```

The result is saved with name `pyslresults-[task name]-[subset name]-[method name].npz` under path to 'non_series' folder of a dataset. You can use `13_metrics_from_saved_results.ipynb` to calculate the metrics.

#### FFN: Feedforward Network

By default the path to the main program is `[RD]/Codes/DeepLearningModels/python/betterlearner.py`.

Use the following command to run feedforward network on specific dataset with fine-tuned hyperparameters:

```bash
python [path to the main program('betterlearner.py')] [name of dataset] [task name] 2 [name of imputed data] [name of fold data] [name of stats of imputed data] --label_type [label type] --static_features_path [path to static features, ‘input.csv’] --static_hidden_dim [2048 for 136 raw features, do not set this for other feature sets] --static_ffn_depth 2 --batch_size 100 --nb_epoch 250 --early_stopping True_BestWeight --early_stopping_patience 20 --batch_normalization True --learning_rate 0.001
```

For example, to run Feedforward Network with 136 raw features from MIMIC-III full dataset on in-hospital mortality prediction task:

```bash
python betterlearner.py mimic3_99p_raw_24h mor 2 imputed-normed-ep_1_24.npz 5-folds.npz imputed-normed-ep_1_24-stdized.npz --label_type 0 --static_features_path ../../../Data/admdata_99p/24hrs_raw/non_series/input.csv --static_hidden_dim 2048 --static_ffn_depth 2 --batch_size 100 --nb_epoch 250 --early_stopping True_BestWeight --early_stopping_patience 20 --batch_normalization True --learning_rate 0.001
```

#### LSTM: LSTM only

By default the path to the main program is `[RD]/Codes/DeepLearningModels/python/betterlearner.py`.

Use the following command to run LSTM on specific dataset with fine-tuned hyperparameters:

```bash
python [path to the main program('betterlearner.py')] [name of dataset] [task name] 1 [name of imputed data] [name of fold data] [name of stats of imputed data] --label_type 0 --without_static --output_dim 2 --batch_size 100 --nb_epoch 250 --early_stopping True_BestWeight --early_stopping_patience 20 --batch_normalization True --learning_rate [0.001 for mortality and icd9 prediction and 0.005 for length of stay prediction] --dropout 0.1
```

For example, to run LSTM with 17 processed features from MIMIC-III full dataset on in-hospital mortality prediction task:

```bash
python betterlearner.py mimic3_17f_24h mor 1 imputed-normed-ep_1_24.npz 5-folds.npz imputed-normed-ep_1_24-stdized.npz --label_type 0 --without_static --output_dim 2 --batch_size 100 --nb_epoch 250 --early_stopping True_BestWeight --early_stopping_patience 20 --batch_normalization True --learning_rate 0.001 --dropout 0.1
```


#### MMDL: Feedforward Network+LSTM

By default the path to the main program is `[RD]/Codes/DeepLearningModels/python/betterlearner.py`.

Use the following command to run Feedforward Network+LSTM on specific dataset with fine-tuned hyperparameters:

```bash
python [path to the main program('betterlearner.py')] [name of dataset] [task name] 1 [name of imputed data] [name of fold data] [name of stats of imputed data] --label_type 0 --ffn_depth 1 --merge_depth 0 --output_dim 2 --batch_size 100 --nb_epoch 250 --early_stopping True_BestWeight --early_stopping_patience 20 --batch_normalization True --learning_rate [0.001 for mortality and icd9 prediction and 0.005 for length of stay prediction] --dropout 0.1
```

For example, to run Feedforward Network+LSTM with 17 processed features from MIMIC-III full dataset on in-hospital mortality prediction task:

```bash
python betterlearner.py mimic3_17f_24h mor 1 imputed-normed-ep_1_24.npz 5-folds.npz imputed-normed-ep_1_24-stdized.npz --label_type 0 --ffn_depth 1 --merge_depth 0 --output_dim 2 --batch_size 100 --nb_epoch 250 --early_stopping True_BestWeight --early_stopping_patience 20 --batch_normalization True --learning_rate 0.001 --dropout 0.1
```

#### Score methods(SAPS-II, Modified SAPS-II and SOFA)

Run `13_get_score-results_firstXhrs_17-features-processed.ipynb` to calculate metrics for score methods.
