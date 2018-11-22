
# Benchmarking on MIMIC-III Dataset

## Reference

Sanjay Purushotham*, Chuizheng Meng*, Zhengping Che, and Yan Liu. "[Benchmarking Deep Learning Models on Large Healthcare Datasets.](https://www.sciencedirect.com/science/article/pii/S1532046418300716)" Journal of Biomedical Informatics (JBI). 2018.

An earlier version is available on arXiv ([arXiv preprint arXiv:1710.08531](https://arxiv.org/abs/1710.08531)).

## Requirements

### Database

You must have the database [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/) running on your local machine or on a server. You have to define the connection to the database in the function `getConnection` in file `utils.py` as follows:

```python
def getConnection():
    return psycopg2.connect("dbname='[name of database]' user='[username]' host='[host name of the machine running the database]' password='[password]' port='[port number]'")
```

### Packages

- psycopg2==2.7.1
- numpy==1.13.1
- scipy==0.19.1
- scikit-learn==0.19.0
- matplotlib==2.0.2
- pandas==0.20.3
- For data preparation and SuperLearner:
    - Anaconda 3==4.4.0
- For Feedforward Network and Feedforward Network+LSTM:
    - Anaconda 2==4.4.0
    - Theano==0.9.0
    - Keras==2.0.6

## Prepare data for benchmarking

Here are the required steps and their output files for getting the data for benchmarking prepared.
- We use `[RD]` to represent the base directory (`Benchmark_MIMIC_III`) of this project.
- We use `[WD]` to represent the working directory (which is where the preprocessing codes are located by default). Currently it is `[RD]/Codes/mimic3_mvcv`.
- We use `[DD]` to represent the data directory, which is the `Data` folder under root directory by default. Currently it is `[RD]/Data`.
- We use `X` to represent the length of time series(unit is hour), which can either be 24 or 48.

### Select admissions and all features

1. `0_createAdmissionList.ipynb`. Select all admissions from TABLE ICUSTAYS and TABLE TRANSFERS. Also collect admissions which are the first admissions of their patients. The list of all admissions is stored in `[WD]/res/admission_ids.npy` and the list of all first-admissions is stored in `[WD]/res/admission_first_ids.npy`.
2. `1_getItemIdList.ipynb`. Select all itemids from TABLE INPUTEVENTS, OUTPUTEVENTS, CHARTEVENTS, LABEVENTS, MICROBIOLOGYEVENTS, PRESCRIPTIONS. The itemids are stored in the file `[WD]/res/itemids.npy` as a dict:
```
    'input': [itemids from INPUTEVENTS]
    'output': [itemids from OUTPUTEVENTS]
    'chart': [itemids from CHARTEVENTS]
    'lab': [itemids from LABEVENTS]
    'microbio': [itemids from MICROBIOLOGYEVENTS]
    'prescript': [itemids from PRESCRIPTIONS]
```
3. `2_filterItemId_input.ipynb, 3_filterItemId_output.ipynb, 4_filterItemId_chart.ipynb, 5_filterItemId_lab.ipynb, 6_filterItemId_microbio.ipynb, 7_filterItemId_prescript.ipynb`. Divide itemids to numeric features/categorical features/ratio features.
    - INPUTEVENTS
        - All itemids belong to numeric features. Itemids and units are stored in `[WD]/res/filtered_input.npy`: `{'id':[itemids], 'unit':[units]}`
    - OUTPUTEVENTS
        - All itemids belong to numeric features. Itemids and units are stored in `[WD]/res/filtered_output.npy`: `{'id':[itemids], 'unit':[units]}`
    - CHARTEVENTS
        - Numeric features' itemids and units are stored in `[WD]/res/filtered_chart_num.npy`: `{'id': [numeric itemids], 'unit': [units of numeric features]}`
        - Categorical features' itemids and units are stored in `[WD]/res/filtered_chart_cate.npy`: `{'id': [categorical itemids], 'unit': [anything, not used, default is None]}`
        - Ratio features' itemids and units are stored in `[WD]/res/filtered_chart_ratio.npy`: `{'id': [ratio itemids], 'unit': [anything, not used, default is None]}`
    - LABEVENTS
        - Numeric features' itemids and units are stored in `[WD]/res/filtered_lab_num.npy`: `{'id': [numeric itemids], 'unit': [units of numeric features]}`
        - Categorical features' itemids and units are stored in `[WD]/res/filtered_lab_cate.npy`: `{'id': [categorical itemids], 'unit': [anything, not used, default is None]}`
        - Ratio features' itemids and units are stored in `[WD]/res/filtered_lab_ratio.npy`: `{'id': [ratio itemids], 'unit': [anything, not used, default is None]}`
    - MICROBIOLOGYEVENTS
        - All itemids belong to categorical features. temids and units are stored in `[WD]/res/filtered_input.npy`: `{'id':[itemids], 'unit':[anything, not used, default is None]}`
    - PRESCRIPTIONS
        - All itemids belong to numeric features. Itemids and units are stored in `[WD]/res/filtered_input.npy`: `{'id':[itemids], 'unit':[units]}`
4. `8_processing.ipynb`. Generate one data file for each admission id from TABLE ICUSTAYS and TABLE TRANSFERS. 
    - The file defining rules of unit conversion `[WD]/config/unitsmap.unit` is needed.
    - The data file for each admission id is stored in `[WD]/admdata/adm-[admission id].npy`.
5. `9_collect_mortality_labels.ipynb`. Generate mortality labels and timestamps. One file for each admission id. The timestamps are stored in `[WD]/admdata_times` and the mortality labels are stored in `[WD]/admdata_timelabels`. The mortality labels(instead of labels generated in `8_processing.ipynb`) will be used for classification.
6. `9_getValidDataset.ipynb`. Collect all data of valid admission ids (first admissions).

### Generate 17 processed features, 17 raw features and 140 raw features

1. Run `run_necessary_sqls.ipynb` to run all necessary sql scripts to get all views prepared.
2. `10_get_17-features-processed.ipynb`. Generate sparse matrices for 17 processed features for first 24hrs and first 48hrs data. The output files `DB_merged_Xhrs.npy`, `ICD9-Xhrs.npy`, `AGE_LOS_MORTALITY_Xhrs.npy`, `ADM_FEATURES_Xhrs.npy`, `ADM_LABELS_Xhrs.npy` are stored in folder `[DD]/admdata_17f/Xhrs/`.
3. `10_get_17-features-raw.ipynb`: Generate sparse matrices for 17 raw features for first 24hrs and first 48hrs data. The output files `DB_merged_Xhrs.npy`, `ICD9-Xhrs.npy`, `AGE_LOS_MORTALITY_Xhrs.npy`, `ADM_FEATURES_Xhrs.npy`, `ADM_LABELS_Xhrs.npy` are stored in folder `[DD]/admdata_17f/Xhrs_raw/`.
4. `10_get_99plus-features-raw.ipynb`: Generate sparse matrices for 140 raw features for first 24hrs and first 48hrs data. The output files `DB_merged_Xhrs.npy`, `ICD9-Xhrs.npy`, `AGE_LOS_MORTALITY_Xhrs.npy`, `ADM_FEATURES_Xhrs.npy`, `ADM_LABELS_Xhrs.npy` are stored in folder `[DD]/admdata_99p/Xhrs_raw/`. The 140 raw features will be filtered according to their missing rates in later scripts so not all of them will be kept. In our experiments we dropped features with missing rates >= 1-5e-4 and kept 136 features.

### Generate time series

1. `11_get_time_series_sample_17-features-processed_Xhrs.ipynb`
    - `normed-ep-ratio.npz`: Data after averaging and before sampling. For 17 processed features, we should use norm-ep-ratio.npz since it includes PaO2/FiO2 ratio. It is stored in `[DD]/admdata_17f/Xhrs/series`.
    - `imputed-normed-ep_T_X.npz`: Data after sampling and imputation. T (hours) is the length of interval of sampling and X (hours) is the length of time series. It is stored in `[DD]/admdata_17f/Xhrs/series`.
    - `5-folds.npz`: Folds file containing indices of each fold. It is stored in `[DD]/admdata_17f/Xhrs/series`.
    - `normed-ep-ratio-stdized.npz`, `imputed-normed-ep_T_X-stdized.npz`: Mean and standard error of features for each fold. It is stored in `[DD]/admdata_17f/Xhrs/series`.
    - For Carevue data and Metavision data, we also generate output files above. The folders are `[DD]/admdata_17f/Xhrs/series/cv`(for Carevue) and `[DD]/admdata_17f/Xhrs/series/mv`(for Metavision).
2. `11_get_time_series_sample_17-features-raw_Xhrs.ipynb`
    - `normed-ep.npz`: Data after averaging and before sampling. It is stored in `[DD]/admdata_17f/Xhrs_raw/series`.
    - `imputed-normed-ep_T_X.npz`: Data after sampling and imputation. T (hours) is the length of interval of sampling and X (hours) is the length of time series. It is stored in `[DD]/admdata_17f/Xhrs_raw/series`.
    - `5-folds.npz`: Folds file containing indices of each fold. It is stored in `[DD]/admdata_17f/Xhrs_raw/series`.
    - `normed-ep-stdized.npz`, `imputed-normed-ep_T_X-stdized.npz`: Mean and standard error of features for each fold. It is stored in `[DD]/admdata_17f/Xhrs_raw/series`.
    - For Carevue data and Metavision data, we also generate output files above. The folders are `[DD]/admdata_17f/Xhrs_raw/series/cv`(for Carevue) and `[DD]/admdata_17f/Xhrs_raw/series/mv`(for Metavision).
    - `tsmean_Xhrs.npz`: Non-temporal data for SuperLearner. It is stored in `[DD]/admdata_17f/Xhrs_raw/non_series`.
3. `11_get_time_series_sample_99plus-features-raw_Xhrs.ipynb`
    - `normed-ep.npz`: Data after averaging and before sampling. It is stored in `[DD]/admdata_99p/Xhrs_raw/series`.
    - `imputed-normed-ep_T_X.npz`: Data after sampling and imputation. T (hours) is the length of interval of sampling and X (hours) is the length of time series. It is stored in `[DD]/admdata_99p/Xhrs_raw/series`.
    - `5-folds.npz`: Folds file containing indices of each fold. It is stored in `[DD]/admdata_99p/Xhrs_raw/series`.
    - `normed-ep-stdized.npz`, `imputed-normed-ep_T_X-stdized.npz`: Mean and standard error of features for each fold. It is stored in `[DD]/admdata_99p/Xhrs_raw/series`.
    - For Carevue data and Metavision data, we also generate output files above. The folders are `[DD]/admdata_99p/Xhrs_raw/series/cv`(for Carevue) and `[DD]/admdata_99p/Xhrs_raw/series/mv`(for Metavision).
    - `tsmean_Xhrs.npz`: Non-temporal data for SuperLearner. It is stored in `[DD]/admdata_99p/Xhrs_raw/non_series`.

### Generate non-temporal features for SuperLearner

1. `12.0_get_severity_scores_firstXhrs_17-features-processed.ipynb`: This script is used for running SQLs calculating non-temporal features in first X hours. It will generate materialized views useful for later processing.
2. `12_get_avg_firstXhrs_17-features-processed(fromdb).ipynb`
    - `input.csv`, `input_cv.csv`, `input_mv.csv`: Non-temporal features. These files are stored in `[DD]/admdata_17f/Xhrs/non_series`.
    - `input_sapsiisubscores.csv`, `input_sapsiisubscores_mv.csv`, `input_sapsiisubscores_cv.csv`: Subscores of SAPS-II used for SuperLearner-I. These files are stored in `[DD]/admdata_17f/Xhrs/non_series`.
    - `output.csv`, `output_cv.csv`, `output_mv.csv`: Mortality labels. These files are stored in `[DD]/admdata_17f/Xhrs/non_series`.
    - `input_F_T.csv`, `output_F_T.csv`: Input files and mortality labels for each fold. F is the rank of fold and T is the rank of mortality task. These files are only used for R version of SuperLearner. These files are stored in `[DD]/admdata_17f/Xhrs/non_series/folds`, `[DD]/admdata_17f/Xhrs/non_series/folds_sapsiiscores`, `[DD]/admdata_17f/Xhrs/non_series/folds/cv` and `[DD]/admdata_17f/Xhrs/non_series/folds_sapsiiscores/cv`.
3. `12_get_avg_firstXhrs_17-features-raw.ipynb`:
    - `input.csv`, `input_cv.csv`, `input_mv.csv`: Non-temporal features. These files are stored in `[DD]/admdata_17f/Xhrs_raw/non_series`.
    - `output.csv`, `output_cv.csv`, `output_mv.csv`: Mortality labels. These files are stored in `[DD]/admdata_17f/Xhrs_raw/non_series`.
    - `input_F_T.csv`, `output_F_T.csv`: Input files and mortality labels for each fold. F is the rank of fold and T is the rank of mortality task. These files are only used for R version of SuperLearner. These files are stored in `[DD]/admdata_17f/Xhrs_raw/non_series/folds`, `[DD]/admdata_17f/Xhrs_raw/non_series/folds/cv`.
4. `12_get_avg_firstXhrs_99plus-features-raw.ipynb`
    - `input.csv`, `input_cv.csv`, `input_mv.csv`: Non-temporal features. These files are stored in `[DD]/admdata_99p/Xhrs_raw/non_series`.
    - `output.csv`, `output_cv.csv`, `output_mv.csv`: Mortality labels. These files are stored in `[DD]/admdata_99p/Xhrs_raw/non_series`.
    - `input_F_T.csv`, `output_F_T.csv`: Input files and mortality labels for each fold. F is the rank of fold and T is the rank of mortality task. These files are only used for R version of SuperLearner. These files are stored in `[DD]/admdata_99p/Xhrs_raw/non_series/folds`, `[DD]/admdata_99p/Xhrs_raw/non_series/folds/cv`.

### Evaluate performance

#### SuperLearner(R version)

Use the following command to run [SuperLearner(R version)](https://cran.r-project.org/package=SuperLearner) on specific dataset:

```bash
Rscript [WD]/r_process_mimic_all.r [path to the folder storing input_F_T.csv and output_F_T.csv, no slash in the end] [rank of label]
```

For example, run SuperLearner-I(which means using subscores as features) with 17 processed features from MIMIC-III full dataset on in-hospital mortality prediction task:

```bash
Rscript [WD]/r_process_mimic_all.r [DD]/admdata_17f/24hrs/non_series/folds 0
```

The result is saved with the name `results_[rank of label].rds` under the same folder with where input files and output files are located. You can use `13_r_validation.ipynb` to parse the result and calculate the metrics.

SuperLearner(R version) only supports mortality prediction tasks since its efficiency is too low and we do not plan to apply it to other tasks.

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
