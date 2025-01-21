# Deep learning-based model for prediction of early recurrence and therapy response on whole slide images in non-muscle-invasive bladder cancer: a retrospective, multicentre study

## Introduction
Background Accurate prediction of early recurrence is essential for disease management of patients with non-muscle-invasive bladder cancer (NMIBC). We aimed to develop and validate a deep learning-based early recurrence predictive model (ERPM) and a treatment response predictive model (TRPM) on whole slide images to assist clinical decision making.

Method In this retrospective, multicentre study, we included consecutive patients with pathology-confirmed NMIBC who underwent transurethral resection of bladder tumour from five centres. Based on multi-instance and ensemble learning, the ERPM was developed to make predictions on haematoxylin and eosin (H&E) staining and immunohistochemistry staining slides.  Sharing the same architecture of the ERPM, the TRPM was trained and evaluated by cross validation on patients who received Bacillus Calmette–Guérin (BCG). The performance of the ERPM was mainly evaluated and compared with the clinical model, H&E-based model, and integrated model through the area under the curve. Survival analysis was performed to assess the prognostic capability of the ERPM.

Findings Between January 1, 2017, and September 30, 2023, 4395 whole slide images of 1275 patients were included to train and validate the models. In validation cohorts, the ERPM was superior to the clinical and H&E-based model in predicting early recurrence (area under the curve: 0.761-0.837 vs 0.649-0.723) and was on par with the integrated model. It also stratified recurrence-free survival significantly (p<0·0001) with a hazard ratio of 4.50 (95% CI 3.10–6.53). The TRPM performed well in predicting BCG-unresponsive NMIBC (accuracy 84.1%), which precisely predicted BCG-unresponsive patients with progression with an accuracy of 100%.

Interpretation The ERPM showed promising performance in predicting early recurrence and recurrence-free survival of patients with NMIBC after surgery and with further validation and in combination with TRPM could be used to guide the management of NMIBC.

## Training

### Training using Train and Test List
To train the model using a train and test list, refer to the script `run_train.sh`.

**Command:**
```bash
bash run_train.sh
```

### Training using File List with Cross-Validation
To train the model using a file list with cross-validation, refer to the script `run_train_fold.sh`.

**Command:**
```bash
bash run_train_fold.sh
```

### Training with Cross-Validation using NNI
To train the model using a file list with cross-validation and NNI (Neural Network Intelligence), refer to the script `run_train_nni.sh` and the configuration file `nni_exp/ss_file/search_space.yaml`.

**Command:**
```bash
nnictl create --config nni_exp/ss_file/search_space.yaml
```

---

## Datasets
### Input Data Structure

You should organize the data according to your specific dataset. Below is an overview of the required input data structure:

- **`input_feats_path`**: Directory containing all the feature files.

- **`input_dataset_file`**: A file containing the list of all target `pathology_no` values.

- **`target_wsi_type`**: Specifies the type of WSI to be used.

### Input PKL Structure (`input_label_info_dict`)

The input data should be organized in a dictionary (`input_label_info_dict`) with the following structure:

- **Key**: `pathology_no`
- **Value**: A list containing:
  - A list of filenames corresponding to the WSIs.
  - A list of feature file paths. (e.g., the file path for "he", "p53", "ki67", "ck20")
  - A value representing the pathology class.

Note that feature file paths should be relative to `input_feats_path`. Therefore, the final feature path should be constructed using `os.path.join(input_feats_path, features_file_path)`.

#### Example:

```python
pathology_no_dict = [tmp_wsi_list, tmp_path_list,  class_value]
```

---

## Testing

### Single Model Inference
For single `.pth` model inference, refer to the script `run_infer.sh`.

**Command:**
```bash
bash run_infer.sh
```

### Ensemble Model Inference (Result from `run_train_fold.sh` or NNI Training)
For ensemble model inference (e.g., the result from `run_train_fold.sh` or NNI training), refer to the script `run_infer_esb.sh`.

**Command:**
```bash
bash run_infer_esb.sh
```

**Input Model CSV File:**
The `input_model_rec_path` parameter in `run_infer_esb.sh` should be a CSV file containing the column `best_model`, which holds the paths of the target models.

For example:
```csv
best_model
path/to/model1
path/to/model2
path/to/model3
path/to/model4
path/to/model5
...
```

### Ensemble Model Inference for a Directory
For ensemble model inference on all models within a directory (typically used after NNI training), refer to the script `run_infer_esb_auto.sh`.

**Command:**
```bash
bash run_infer_esb_auto.sh
```

### Inference Without Labels (No Label Required)
For inference without labels (i.e., `class_value` in `pathology_no_dict` can be any value), refer to the script `run_infer_esb_no_label.sh`. This is similar to `run_infer_esb.sh`, but does not require label values.

**Command:**
```bash
bash run_infer_esb_no_label.sh
```


