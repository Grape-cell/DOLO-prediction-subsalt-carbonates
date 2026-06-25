# Core-Calibrated DOLO Prediction from Well Logs

This repository provides Python scripts and Excel data templates for core-calibrated dolomite content (DOLO) prediction in subsalt carbonate reservoirs using conventional well logs. The workflow includes LDA-based well-log curve selection, multi-model regression, leave-one-well-out validation, independent-well testing, SHAP interpretation, and feature-ablation analysis.

This repository supports the manuscript:

**Core-Calibrated Prediction of Dolomite Content in Subsalt Rapid-Facies-Transition Carbonate Reservoirs Using Mud-Logging Data: A Case Study of the Majiagou Formation, Ordos Basin**

## 1. Repository contents

| File | Description |
| --- | --- |
| `LDA-Based Well Log Curve Selection.py` | Performs LDA-based screening of candidate well-log curves and exports class predictions, feature-contribution statistics, and diagnostic figures. |
| `Regression Prediction of Dolomite Mineral Content.py` | Trains and evaluates regression models for DOLO prediction, including leave-one-well-out validation, independent-well testing, stacking, classification diagnostics, SHAP interpretation, and feature-ablation analysis. |
| `LDA.xlsx` | Example or template dataset for LDA-based well-log curve selection. |
| `training_set.xlsx` | Example or template training dataset for regression-model development. |
| `prediction_set.xlsx` | Example or template independent prediction-well dataset for generalization testing. |
| `requirements.txt` | Recommended Python dependencies. |
| `LICENSE` | Open-source license for this repository. |

## 2. Software environment

The scripts were developed for Python 3.10 or later. Python 3.12 is also suitable if all required packages are installed.

Create and activate a clean environment:

```bash
py -m venv .venv
.venv\\Scripts\\activate
```

Install required packages:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xgboost lightgbm shap
```

If `python` points to the Microsoft Store placeholder on Windows, use `py` instead of `python` when creating the environment.

## 3. Input data requirements

Input files should be prepared in Excel format. Column names must match the script settings exactly.

### 3.1 LDA screening dataset

`LDA-Based Well Log Curve Selection.py` reads `LDA.xlsx` with the following required columns:

```text
Depth, DOLO, Pe, DEN, CNL, AC, GR, RLLD, RLLS
```

The script uses `Depth` as the depth index, `DOLO` as the dolomite-content target, and the following candidate logging curves:

```python
feature_cols = ['Pe', 'DEN', 'CNL', 'AC', 'GR', 'RLLD', 'RLLS']
```

DOLO is discretized into three classes:

| Class | DOLO range |
| --- | --- |
| Low dolomite | 0-30% |
| Medium dolomite | 30-70% |
| High dolomite | 70-100% |

Rows containing missing values in required columns are removed before LDA fitting.

### 3.2 Regression datasets

`Regression Prediction of Dolomite Mineral Content.py` reads `training_set.xlsx` and `prediction_set.xlsx`.

The training dataset requires:

```text
Well, Depth, AC, CNL, DEN, DOLO, GR, PE
```

The prediction dataset requires:

```text
Depth, AC, CNL, DEN, DOLO, GR, PE
```

The regression workflow uses the following LDA-selected input curves:

```python
feature_cols = ['CNL', 'DEN', 'AC', 'PE', 'GR']
```

Note that the photoelectric absorption factor is named `Pe` in `LDA.xlsx` but `PE` in the regression datasets. The supplied scripts already match these file-specific column names.

For leave-one-well-out validation, the `Well` column in `training_set.xlsx` must contain at least two different well names. Three or more wells are recommended.

## 4. Running the workflow

Run the scripts from the repository directory.

### 4.1 LDA-based well-log curve selection

```bash
python "LDA-Based Well Log Curve Selection.py"
```

Main outputs include:

| Output file | Description |
| --- | --- |
| `LDA_all_data_classified.xlsx` | Measured DOLO class, LDA-predicted class, class probabilities, and feature-contribution results. |
| `LDA_all_data_LDA_projection.jpg` | Two-dimensional LDA projection if two discriminant axes are available. |
| `LDA_all_data_LDA_hist.jpg` | One-dimensional LDA histogram if only one discriminant axis is available. |
| `LDA_all_data_feature_importance.jpg` | Relative contribution of candidate well-log curves based on LDA coefficients. |
| `LDA_all_data_depth_profile.jpg` | Depth-domain comparison between measured and LDA-predicted dolomite classes. |

### 4.2 Regression prediction of dolomite content

Before running the regression script, check the output directory in:

```python
output_dir = r"..."
```

For local testing, a simple relative path is recommended:

```python
output_dir = r".\\outputs"
```

Then run:

```bash
python "Regression Prediction of Dolomite Mineral Content.py"
```

The regression script performs:

1. Data loading, cleaning, and standardization.
2. Leave-one-well-out validation using `GroupKFold`.
3. Training of XGBoost, random forest, and LightGBM regressors.
4. Independent prediction-well evaluation using `R2`, `RMSE`, and `MAE`.
5. Leakage-controlled stacking based on grouped out-of-fold predictions.
6. Classification-based diagnostics for low-, medium-, and high-dolomite intervals.
7. Residual analysis along depth.
8. SHAP-based feature interpretation.
9. Feature-ablation experiments.

Main numerical output:

| Output file | Description |
| --- | --- |
| `SCI_DOLO_NB.xlsx` | Main numerical output. The `Prediction` sheet contains measured DOLO, model predictions, best-model prediction, stacking prediction, and uncertainty estimates. The `Metrics` sheet contains `R2`, `RMSE`, and `MAE`. |

Main figure outputs include:

| Output file pattern | Description |
| --- | --- |
| `Model_Compare.jpg` | Comparison of model `R2` values on the prediction well. |
| `Error_Depth.jpg` | Residual distribution along depth for the best model. |
| `SCI_SHAP_summary.jpg` | SHAP summary plot for the best model. |
| `SCI_SHAP_bar.jpg` | Mean absolute SHAP importance plot. |
| `SCI_ROC_XGB.jpg`, `SCI_ROC_RF.jpg`, `SCI_ROC_LGB.jpg` | Class-wise ROC curves. |
| `SCI_CM_*.jpg` | Confusion matrices for regression-derived dolomite classes. |
| `SCI_CrossPlot_*.jpg` | Measured-versus-predicted DOLO crossplots. |
| `SCI_Residual_*.jpg` | Residual histograms for each model. |
| `SCI_Pairplot.jpg` | Pairwise feature-distribution plot for the prediction well. |

## 5. Feature-ablation experiments

Feature-ablation experiments can be performed by manually editing the selected input-curve list in `Regression Prediction of Dolomite Mineral Content.py`.

The default feature list is:

```python
feature_cols = ['CNL', 'DEN', 'AC', 'PE', 'GR']
```

To remove a single curve, delete the corresponding item from `feature_cols`. For example, to test the model without `PE`, use:

```python
feature_cols = ['CNL', 'DEN', 'AC', 'GR']
```

To test the model without the `DEN` + `AC` group, use:

```python
feature_cols = ['CNL', 'PE', 'GR']
```

Each ablation run should be saved in a separate output directory so that `R2`, `RMSE`, `MAE`, residual statistics, class-based diagnostics, and SHAP outputs can be compared with the full-feature baseline.

## 6. Reproducibility check

A complete test run should execute the scripts in the following order:

```bash
python "LDA-Based Well Log Curve Selection.py"
python "Regression Prediction of Dolomite Mineral Content.py"
```

The run is considered successful if:

1. `LDA_all_data_classified.xlsx` is generated.
2. LDA diagnostic figures are generated.
3. `SCI_DOLO_NB.xlsx` is generated in the configured `output_dir`.
4. Leave-one-well-out validation and prediction-well metrics are printed in the terminal.
5. SHAP, ROC, confusion-matrix, crossplot, residual, and model-comparison figures are generated.

## 7. Common issues

| Issue | Likely cause | Solution |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'openpyxl'` | Excel-reading dependency is missing. | Run `python -m pip install openpyxl`. |
| `ModuleNotFoundError: No module named 'xgboost'`, `lightgbm`, or `shap` | Machine-learning dependency is missing. | Install the dependency list shown in Section 2. |
| `ValueError: Missing required columns` | Column names in Excel files do not match script settings. | Check capitalization, especially `Pe` in `LDA.xlsx` and `PE` in regression datasets. |
| Output files are written to an unexpected path | `output_dir` points to a hard-coded directory. | Replace `output_dir` with a valid local path. |
| Garbled Chinese characters appear in the terminal | Windows console encoding issue. | This does not affect model fitting. Use an English-only output path if needed. |
| `Times New Roman` warnings appear during plotting | Font is unavailable to Matplotlib. | Install Times New Roman or allow Matplotlib to use its fallback font. |

## 8. Methodological notes

The LDA stage is used as a supervised screening procedure to evaluate how strongly candidate well-log curves discriminate low-, medium-, and high-dolomite intervals. The regression stage then uses the selected curves to predict continuous DOLO values.

Grouped validation is applied at the well level to reduce information leakage between training and validation samples. The independent prediction-well test provides a stricter estimate of cross-well generalization than random sample splitting.

The final outputs support both model evaluation and geological interpretation. Numerical metrics quantify prediction accuracy, residual-depth plots reveal depth-dependent errors, SHAP values identify feature contributions, and ROC/confusion-matrix plots assess whether continuous DOLO predictions preserve low-, medium-, and high-dolomite classes.

## 9. Data availability

Data will be made available on request. The Excel files provided in this repository are example or template datasets used to demonstrate the workflow and file structure, and do not represent the complete raw dataset used in the manuscript.

## 10. License

This repository is released under the MIT License.

## 11. Contact

For questions about the code or manuscript, please contact:

Fuhao Zheng  
China University of Petroleum (Beijing)
