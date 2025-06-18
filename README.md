# Child Mind Institute: Problematic Internet Use Prediction

This repository contains the code for a machine learning solution to predict problematic internet use based on various participant data and actigraphy time series. The solution employs an ensemble model consisting of LightGBM, XGBoost, CatBoost, and TabNet, trained using a K-Fold cross-validation strategy with optimized thresholds for the evaluation metric.

**Dataset Source:**
The dataset used in this project is from the Kaggle competition:
[Child Mind Institute - Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use)

## Project Structure

* **`main.ipynb` (or similar notebook file):** The primary Jupyter Notebook containing all the code for data loading, preprocessing, feature engineering, model training, and prediction.
* **`trained_voting_model.pkl`:** (Will be generated after running the notebook) This file stores the trained ensemble model for future inference without retraining.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Download the dataset:**
    Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) and place the `train.csv`, `test.csv`, `sample_submission.csv`, `series_train.parquet`, and `series_test.parquet` files into a `kaggle/input/child-mind-institute-problematic-internet-use/` directory within your project structure.

    Your directory structure should look something like this:
    ```
    .
    ├── your_notebook.ipynb
    ├── trained_voting_model.pkl (will be created)
    └── kaggle
        └── input
            └── child-mind-institute-problematic-internet-use
                ├── train.csv
                ├── test.csv
                ├── sample_submission.csv
                ├── series_train.parquet
                └── series_test.parquet
    ```

3.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    (You'll need to create `requirements.txt` by running `pip freeze > requirements.txt` after installing the libraries mentioned in the notebook).

    **Manual Installation (if `requirements.txt` is not available):**
    ```bash
    !pip install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl # This path is specific to Kaggle, you might need to download the wheel file locally or install from PyPI: pip install pytorch-tabnet
    pip install numpy pandas seaborn tqdm scipy polars matplotlib keras tensorflow torch lightgbm xgboost catboost scikit-learn colorama ipython
    ```
    *Note: For `pytorch_tabnet`, you might need to install it from PyPI (`pip install pytorch-tabnet`) if you're not in a Kaggle environment, or download the wheel file if a specific version is required.*
    *Ensure you have a GPU set up for `pytorch_tabnet`, `xgboost` (if using `gpu_hist`), and `catboost` (if `task_type='GPU'`) if you want to leverage GPU acceleration.*

## Code Description

This project implements a comprehensive machine learning pipeline, detailed below:

### 1. Data Loading and Initial Exploration

* **`train.csv`, `test.csv`, `sample_submission.csv`**: Loaded using Pandas.
* **`series_train.parquet`, `series_test.parquet`**: Actigraphy time-series data loaded using Polars for efficient processing.
* Initial data analysis includes:
    * Checking missing values distribution.
    * Comparing `PCIAT-PCIAT_Total` nullity with the target `sii` nullity.
    * Identifying columns present in train but missing in test.
    * Visualizing `Basic_Demos-Enroll_Season`, `Basic_Demos-Sex`, and `Basic_Demos-Age` distributions.
    * Analyzing the target (`sii`) distribution across sexes.
    * Calculating and visualizing correlations between various features and `PCIAT-PCIAT_Total`.

### 2. Actigraphy Data Processing (`analyze_actigraphy` function)

* This function is used for **exploratory data analysis (EDA)** of individual actigraphy time series.
* It reads a specific participant's `.parquet` file.
* Calculates a continuous 'day' representation.
* Derives new features like `diff_seconds` (time difference between readings) and `norm` (magnitude of 3D acceleration).
* Allows filtering for `only_one_week` and excluding `non-wear_flag` data.
* Generates scatter plots of various actigraphy features (X, Y, Z acceleration, ENMO, anglez, light, non-wear flag) over time, providing insights into activity patterns.

### 3. Time Series Feature Engineering (`process_file`, `load_time_series`, `AutoEncoder`, `perform_autoencoder`)

* **`process_file(filename, directory)`**:
    * Reads individual participant's actigraphy `.parquet` files.
    * Drops the `step` column.
    * Calculates descriptive statistics (`.describe()`) for all remaining columns and flattens them into a single array.
    * **New:** Extracts **Frequency Domain Features** using Fast Fourier Transform (`fft`). It calculates the mean magnitude of FFT for each feature in the time series.
    * Concatenates descriptive statistics with FFT features.
* **`load_time_series(dirname)`**:
    * Orchestrates the parallel loading and processing of all actigraphy files using `ThreadPoolExecutor` and `tqdm` for progress tracking.
    * Combines the extracted statistics and FFT features into a single Pandas DataFrame.
* **`AutoEncoder` class (PyTorch)**:
    * A simple deep learning autoencoder architecture used for dimensionality reduction of the time-series features. It consists of an encoder (downsamples to `encoding_dim`) and a decoder (reconstructs the input).
* **`perform_autoencoder(df, encoding_dim, epochs, batch_size)`**:
    * Scales the input time-series statistics using `StandardScaler`.
    * Trains the `AutoEncoder` model using Mean Squared Error (MSE) loss and Adam optimizer.
    * After training, it uses the encoder part to transform the scaled time-series statistics into a lower-dimensional representation (`encoded_data`).
    * Returns the encoded data as a new Pandas DataFrame.

### 4. Main Data Preprocessing and Feature Engineering (`feature_engineering`)

* Loads `train.csv` and `test.csv`.
* Processes time-series data for both train and test sets using the `load_time_series` and `perform_autoencoder` functions.
* Merges the encoded time-series features back with the main `train` and `test` DataFrames based on `id`.
* **`KNNImputer`**: Fills missing numerical values using K-Nearest Neighbors imputation. This is applied *before* the main feature engineering.
* **`feature_engineering(df)`**:
    * Drops all 'Season' columns (as they might have been handled by other means or deemed less important after initial EDA).
    * Creates several new interaction and ratio features, such as `BMI_Age`, `Internet_Hours_Age`, `BFP_BMI`, `Muscle_to_Fat`, `Hydration_Status`, etc., combining existing numerical features to capture more complex relationships.
* Drops rows in the `train` set where the target `sii` is missing.
* Handles infinite values (`np.inf`, `-np.inf`) by converting them to `NaN` to prevent errors in subsequent steps.
* **Categorical Feature Encoding:** The `update` function (which was present in your previous snippet but not explicitly in the provided final version, but the `create_mapping` loop handles it) fills 'Season' NaN values with 'Missing' and converts them to categorical type. Then, `create_mapping` converts these categorical features into integer labels.

### 5. Model Training and Evaluation (`TrainML`)

* **`quadratic_weighted_kappa`**: Implements the Cohen's Kappa score with quadratic weights, which is the evaluation metric for the competition.
* **`threshold_Rounder`**: A utility function to convert continuous model predictions into discrete target labels (0, 1, 2, 3) based on learned thresholds.
* **`evaluate_predictions`**: Used by the optimizer to find the best thresholds by minimizing the negative Kappa score.
* **`TrainML(model_class, test_data)`**:
    * Implements a K-Fold cross-validation strategy using `StratifiedKFold` to maintain the distribution of the target variable across folds.
    * For each fold:
        * Splits data into training and validation sets.
        * Clones the `model_class` (which is your `VotingRegressor` instance).
        * Fits the model on the training data.
        * Makes predictions on both training and validation sets.
        * Stores out-of-fold (OOF) predictions (`oof_non_rounded`).
        * Calculates and prints Quadratic Weighted Kappa (QWK) for training and validation sets.
        * Accumulates test predictions for averaging.
    * After cross-validation, it performs **Kappa Optimization**:
        * Uses `scipy.optimize.minimize` (Nelder-Mead method) to find optimal thresholds (0.5, 1.5, 2.5 initially) that maximize the QWK on the OOF predictions.
    * Applies the optimized thresholds to the OOF predictions and calculates the final optimized QWK.
    * Averages the predictions across all folds for the test set.
    * Applies the optimized thresholds to the averaged test predictions.
    * Creates the final `submission` DataFrame.
    * **Returns the `submission` DataFrame.**
    * **Crucially, the `model` object from the *last fold* is the one that is implicitly retained and would need to be explicitly returned if you want to save a single trained model instance (as implemented in the solution below).**

### 6. Model Definition

* **LightGBM (`LGBMRegressor`)**: Configured with optimized hyperparameters (learning rate, depth, leaves, feature/bagging fractions, L1/L2 regularization) and uses CPU for device.
* **XGBoost (`XGBRegressor`)**: Configured with optimized hyperparameters (learning rate, depth, estimators, subsample, colsample_bytree, L1/L2 regularization) and uses `gpu_hist` for GPU acceleration.
* **CatBoost (`CatBoostRegressor`)**: Configured with optimized hyperparameters (learning rate, depth, iterations, L2 regularization) and uses GPU for task type.
* **`TabNetWrapper`**: A custom wrapper class for `TabNetRegressor` to make it compatible with scikit-learn's API (`BaseEstimator`, `RegressorMixin`).
    * Handles missing values internally using `SimpleImputer`.
    * Splits data for internal validation during TabNet training.
    * Uses `TabNetPretrainedModelCheckpoint` to save the best TabNet model during its training phase.
    * Handles deepcopy for scikit-learn compatibility.
* **`TabNetPretrainedModelCheckpoint`**: A custom callback for `TabNet` training to save the best model based on a monitored metric (`valid_mse`).
* **`VotingRegressor`**: The final ensemble model that combines the predictions from LightGBM, XGBoost, CatBoost, and TabNet.

## How to Run

1.  Ensure all dependencies are installed.
2.  Download the dataset as described in "Setup and Installation".
3.  Open the Jupyter Notebook (`main.ipynb` or your chosen name).
4.  Run all cells in sequence.

The script will:
1. Load and preprocess the data.
2. Perform extensive feature engineering, including time-series aggregation and autoencoder-based dimensionality reduction.
3. Train the `VotingRegressor` using Stratified K-Fold cross-validation.
4. Optimize prediction thresholds based on the Quadratic Weighted Kappa metric.
5. Generate predictions on the test set.
6. **Save the final trained `VotingRegressor` model as `trained_voting_model.pkl`.**
7. Display the submission DataFrame.

---
