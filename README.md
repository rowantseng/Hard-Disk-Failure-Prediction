# Hard Disk Failure Prediction

In this repo, a disk failure prediction model is trained on SMART data.

## Prepro

Data are collected from [Backblaze](https://www.backblaze.com/b2/hard-drive-test-data.html) over 2015-2017. The HDD model `ST4000DM000` is selected because it occupies a large proportion in the provided data. 

There are total 95 columns in the original dataframe including SMART data and disk information. In the first-stage preprocessing, only raw columns are retrieved and normalized columns are dropped. Numerical columns are filled in zeros. In this experiment, `n_ok_delta` is set at 0 because many non-failed samples can be collected from the failed disks before they crashed. Also, broken disks constitute only ~7% of all disks. However, you can replace `n_ok_delta` greater than zero to accommodate more disks in the dataset. The final size of the dataframe is `(31414293, 49)`. You can execute the code as below:

```shell
cd prepro

# Retrieve raw smart features
python clean_df.py
```

Then, we got statistics as below:

```shell
# Distribution of non-failed/failed disks
Num of disks: 36700
Num of failed disks: 2587
Percentage of broken disks: 7.05%

# `n_ok_delta` is set at 0 to get the same number of non-failed disks as failed disks
Num of failed disks: 2587
Num of OK disks: 2587
Size of final dataframe: (3579161, 49) --> Dataset size reduction: 88%
```

In the second-stage preprocessing, failed samples and non-failed samples are prepared evenly. You can execute the code as below:

```shell
python combine_feature.py
```

First, `sample_days=5` and `predict_failure_days=5` are defined. `sample_days` means how many days we want to have in each sample. `predict_failure_days` means how many days before the failure day we treat as “fails soon“ days. Sequences of data that are shorter than `sample_days` are removed. Then, 
select failed samples and non-failed samples(both classes should be balanced). The failed and non-failed samples have below definition. 

```
- failed samples: for which we know they finally failed
- non-failed samples(random selection to ensure better variability in data)
    - samples related to disks that never failed
    - samples related to disks that will fail but "later"
```

In total, we have 127280 data over 5132 unique disks. 

## Train and Evaluate Model Using Scikit-Learn

Please see `hard_drive_failure_prediction_colab.ipynb`. There are full instructions in the notebook. In addtion, the preprocessed data is linked via Google Drive.

Finally, we have training dataframe with size `(102000, 55)` over 4105 disks and testing dataframe with `(25280, 55)` over 1027 disks. The model architecture is `Random Forest`. Training is conducted under cross validation with `split=3`. Testing accuracy is `0.783`. 

The best estimator has the params:
```
{'criterion': 'entropy', 'max_features': 'auto', 'min_samples_leaf': 1, 'n_estimators': 200}
```

## Appendix

### Total Provided Features in the Original Dataframe

```
'date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'smart_1_normalized', 'smart_1_raw', 'smart_2_normalized', 'smart_2_raw', 'smart_3_normalized', 'smart_3_raw', 'smart_4_normalized', 'smart_4_raw', 'smart_5_normalized', 'smart_5_raw', 'smart_7_normalized', 'smart_7_raw', 'smart_8_normalized', 'smart_8_raw', 'smart_9_normalized', 'smart_9_raw', 'smart_10_normalized', 'smart_10_raw', 'smart_11_normalized', 'smart_11_raw', 'smart_12_normalized', 'smart_12_raw', 'smart_13_normalized', 'smart_13_raw', 'smart_15_normalized', 'smart_15_raw', 'smart_22_normalized', 'smart_22_raw', 'smart_183_normalized', 'smart_183_raw', 'smart_184_normalized', 'smart_184_raw', 'smart_187_normalized', 'smart_187_raw', 'smart_188_normalized', 'smart_188_raw', 'smart_189_normalized', 'smart_189_raw', 'smart_190_normalized', 'smart_190_raw', 'smart_191_normalized', 'smart_191_raw', 'smart_192_normalized', 'smart_192_raw', 'smart_193_normalized', 'smart_193_raw', 'smart_194_normalized', 'smart_194_raw', 'smart_195_normalized', 'smart_195_raw', 'smart_196_normalized', 'smart_196_raw', 'smart_197_normalized', 'smart_197_raw', 'smart_198_normalized', 'smart_198_raw', 'smart_199_normalized', 'smart_199_raw', 'smart_200_normalized', 'smart_200_raw', 'smart_201_normalized', 'smart_201_raw', 'smart_220_normalized', 'smart_220_raw', 'smart_222_normalized', 'smart_222_raw', 'smart_223_normalized', 'smart_223_raw', 'smart_224_normalized', 'smart_224_raw', 'smart_225_normalized', 'smart_225_raw', 'smart_226_normalized', 'smart_226_raw', 'smart_240_normalized', 'smart_240_raw', 'smart_241_normalized', 'smart_241_raw', 'smart_242_normalized', 'smart_242_raw', 'smart_250_normalized', 'smart_250_raw', 'smart_251_normalized', 'smart_251_raw', 'smart_252_normalized', 'smart_252_raw', 'smart_254_normalized', 'smart_254_raw', 'smart_255_normalized', 'smart_255_raw'
```