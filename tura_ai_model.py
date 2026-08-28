"""
TURA-GeoSync AI Modeling Pipeline
Predicts Earth's Length of Day (LOD) variations from regional tectonic lithospheric mass anomalies
using signal filtering, feature engineering, and Random Forest regression.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    DEFAULT_GRACE_NC_PATH,
    REGIONS,
    RANDOM_STATE,
    TEST_SIZE,
    RF_N_ESTIMATORS
)
from iers_lod import fetch_lod_from_paris_obs
from grace_litosfer import process_grace_lithosphere
from sync_data import synchronize_lod_to_grace
from correlation_model import extract_regional_mass_anomaly


def load_and_sync_dataset(nc_file_path: Optional[Path] = None, region_key: str = "himalayas") -> pd.DataFrame:
    """
    Ingests IERS LOD and GRACE lithosphere datasets, synchronizes timestamps, and extracts regional mass anomalies.

    Parameters
    ----------
    nc_file_path : Optional[Path]
        Path to GRACE NetCDF dataset file. Defaults to config.DEFAULT_GRACE_NC_PATH.
    region_key : str
        Region key defined in config.REGIONS. Defaults to 'himalayas'.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'LOD' and 'Mass' anomaly time series.
    """
    path = nc_file_path or DEFAULT_GRACE_NC_PATH
    print("[Pipeline Phase 1] Data Ingestion & Regional Extraction...")

    grace_da = process_grace_lithosphere(path)
    lod_df = fetch_lod_from_paris_obs()

    if grace_da is None or lod_df is None:
        raise ValueError("Failed to load required input datasets for pipeline.")

    synced_lod = synchronize_lod_to_grace(lod_df, grace_da)

    reg_cfg = REGIONS.get(region_key, REGIONS["himalayas"])
    mass_anomaly = extract_regional_mass_anomaly(
        grace_da,
        lat_min=reg_cfg["lat_min"],
        lat_max=reg_cfg["lat_max"],
        lon_min=reg_cfg["lon_min"],
        lon_max=reg_cfg["lon_max"]
    )

    combined_df = pd.DataFrame({'LOD': synced_lod['LOD'], 'Mass': mass_anomaly})
    return combined_df


def apply_signal_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies median rolling filter to LOD noise (leap second noise)
    and moving average filter to Mass anomaly series (monsoonal seasonal noise).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw 'LOD' and 'Mass'.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'LOD_Clean' and 'Mass_Clean' columns.
    """
    print("[Pipeline Phase 2] Applying Signal Filtering (Rolling Window Filters)...")
    cleaned_df = df.copy()

    # 3-month median filter for LOD high-frequency noise
    cleaned_df['LOD_Clean'] = cleaned_df['LOD'].rolling(window=3, center=True).median()

    # 12-month moving average filter for mass anomaly seasonal monsoon cycle
    cleaned_df['Mass_Clean'] = cleaned_df['Mass'].rolling(window=12, center=True).mean()

    return cleaned_df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers memory (lagged features) and velocity (first-derivative) dynamics for AI training.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame containing 'LOD_Clean' and 'Mass_Clean'.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame with all NaNs dropped.
    """
    print("[Pipeline Phase 3] Engineering Temporal Lag & Velocity Features...")
    feat_df = df.copy()

    feat_df['Mass_Lag1'] = feat_df['Mass_Clean'].shift(1)
    feat_df['Mass_Lag3'] = feat_df['Mass_Clean'].shift(3)
    feat_df['Mass_Lag6'] = feat_df['Mass_Clean'].shift(6)
    feat_df['Mass_Velocity'] = feat_df['Mass_Clean'].diff()
    feat_df['LOD_Lag1'] = feat_df['LOD_Clean'].shift(1)

    return feat_df.dropna()


def train_random_forest_model(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: str = 'LOD_Clean'
) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, Dict[str, float]]:
    """
    Trains an optimized Random Forest Regressor using chronological split (no shuffling).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    feature_cols : Optional[list]
        List of feature column names. Defaults to standard feature set.
    target_col : str
        Target column name. Defaults to 'LOD_Clean'.

    Returns
    -------
    Tuple containing trained model, X_train, X_test, y_train, y_test, y_pred, and evaluation metrics dict.
    """
    print("[Pipeline Phase 4] Training Random Forest Machine Learning Model...")

    if feature_cols is None:
        feature_cols = ['Mass_Clean', 'Mass_Lag1', 'Mass_Lag3', 'Mass_Lag6', 'Mass_Velocity', 'LOD_Lag1']

    X = df[feature_cols]
    y = df[target_col]

    # Chronological train-test split (preserves time-series sequence)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))

    metrics = {'R2': r2, 'MSE': mse}

    print("-" * 50)
    print("RANDOM FOREST MODEL PERFORMANCE EVALUATION")
    print(f"R-squared Score (Variance Explained): {r2:.4f}")
    print(f"Mean Squared Error (MSE):           {mse:.6e}")
    print("-" * 50)

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


def plot_model_results(y_test: pd.Series, y_pred: np.ndarray, region_label: str = "Himalaya Tectonics") -> None:
    """
    Visualizes ground truth vs Random Forest predictions.

    Parameters
    ----------
    y_test : pd.Series
        Actual LOD test set ground truth values.
    y_pred : np.ndarray
        Model predicted LOD values.
    region_label : str
        Region label for chart title.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label="Observed LOD Anomaly", color="tab:red", linewidth=2)
    plt.plot(y_test.index, y_pred, label="Random Forest Prediction", color="tab:green", linestyle="--", linewidth=2)
    plt.title(f"TURA-GeoSync AI: Earth Rotation (LOD) Prediction via {region_label}", fontsize=14, fontweight='bold')
    plt.xlabel("Date (Test Period)")
    plt.ylabel("LOD Anomaly (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_tura_ai_pipeline(nc_file_path: Optional[Path] = None, show_plot: bool = True) -> Dict[str, Any]:
    """
    Executes end-to-end TURA-GeoSync machine learning pipeline.

    Returns
    -------
    Dict[str, Any]
        Pipeline execution dictionary containing model, predictions, and metrics.
    """
    print("==================================================")
    print("Starting TURA-GeoSync AI Pipeline (Random Forest)")
    print("==================================================\n")

    # 1. Ingestion & Synchronization
    raw_df = load_and_sync_dataset(nc_file_path)

    # 2. Noise Filtering
    filtered_df = apply_signal_filtering(raw_df)

    # 3. Feature Engineering
    engineered_df = engineer_time_features(filtered_df)

    # 4. ML Model Training & Evaluation
    model, X_tr, X_te, y_tr, y_te, y_pred, metrics = train_random_forest_model(engineered_df)

    # 5. Visualization
    if show_plot:
        plot_model_results(y_te, y_pred)

    return {
        "model": model,
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
        "y_pred": y_pred,
        "metrics": metrics
    }


if __name__ == "__main__":
    run_tura_ai_pipeline()