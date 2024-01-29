
### Preparing Data


- **Download Raw Data**

    You can download PEMS08 and METR-LA at [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp) .
    NYC_Taxi is available in: https://pan.baidu.com/s/1usoSQtzUk8JDh-v0B7f52g with the password kyt9. 
    Download Solar datasets from https://github.com/laiguokun/multivariate-time-series-data.
    Unzip them to `datasets/raw_data/`

- **Pre-process Data**

    ```bash
    cd /path/to/your/project
    python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py
    ```

    Replace `${DATASET_NAME}` with your dataset. The processed data will be placed in `datasets/${DATASET_NAME}`.


### Train

- **Pre-training Stage**

    ```bash
    python step/run.py --cfg='model/MAEST_$DATASET.py' --gpus '0'
    ```
  Replace `$DATASET_NAME` with with your dataset.
- **Forecasting Stage** 
    After pre-training TSFormer, move your pre-trained best checkpoint to `MAEST_ckpt/` and rename `MAEST_$DATASET_NAME.pt`.
    Then
    ```bash
      python step/run.py --cfg='model/enhance_$DATASET.py' --gpus='0'
    ```
    Replace `$DATASET_NAME` with with your dataset.

The code is developed with [BasicTS](https://github.com/zezhishao/BasicTS).

