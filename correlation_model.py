"""
Regional Mass Anomaly & Pearson Correlation Analysis Module
Extracts spatially averaged mass anomalies for specific tectonic regions (bounding box)
and computes statistical correlation against Length of Day (LOD) variations.
"""

from typing import Tuple, Optional
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from config import REGIONS
from iers_lod import fetch_lod_from_paris_obs
from grace_litosfer import process_grace_lithosphere
from sync_data import synchronize_lod_to_grace


def extract_regional_mass_anomaly(
    grace_da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> pd.Series:
    """
    Extracts spatial mean mass anomaly time series for a specified geographic bounding box.

    Parameters
    ----------
    grace_da : xr.DataArray
        Filtered GRACE mass anomaly DataArray.
    lat_min, lat_max : float
        Latitude range (degrees North).
    lon_min, lon_max : float
        Longitude range (degrees East).

    Returns
    -------
    pd.Series
        1D time series of spatially averaged mass anomalies.
    """
    print(f"[Correlation] Isolating regional mass anomaly (Lat: [{lat_min}, {lat_max}], Lon: [{lon_min}, {lon_max}])...")

    # Spatial slicing using xarray coordinate selection
    region = grace_da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Compute spatial mean across latitude and longitude dimensions lazily, converting to pandas Series
    region_timeseries = region.mean(dim=['lat', 'lon']).to_series()

    return region_timeseries


def plot_correlation(
    synced_lod_df: pd.DataFrame,
    regional_mass_series: pd.Series,
    region_name: str = "Himalaya & Tibetan Plateau",
    show_plot: bool = True
) -> Tuple[float, float]:
    """
    Computes Pearson correlation coefficient and renders a dual-axis time-series visualization.

    Parameters
    ----------
    synced_lod_df : pd.DataFrame
        Synchronized LOD DataFrame containing 'LOD' column.
    regional_mass_series : pd.Series
        Spatially averaged regional mass anomaly series.
    region_name : str
        Human-readable name of the tectonic region.
    show_plot : bool
        Whether to render interactive plot display.

    Returns
    -------
    Tuple[float, float]
        Pearson correlation coefficient (r) and p-value.
    """
    print(f"\n[Correlation] Analyzing correlation for region: {region_name}...")

    # Combine LOD and Mass Anomaly into a single DataFrame and drop unmatched NaNs
    df = pd.DataFrame({
        'LOD_Anomaly': synced_lod_df['LOD'],
        'Mass_Anomaly': regional_mass_series
    }).dropna()

    # Compute Pearson Correlation
    r_coeff, p_val = pearsonr(df['LOD_Anomaly'], df['Mass_Anomaly'])

    print("-" * 50)
    print(f"Region: {region_name}")
    print(f"Pearson Correlation Coefficient (r): {r_coeff:.4f}")
    print(f"Statistical Significance (p-value): {p_val:.4e}")
    print("-" * 50)

    if show_plot:
        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Primary Axis: LOD Anomaly
        color1 = 'tab:red'
        ax1.set_xlabel('Date (Year)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('LOD Anomaly (ms)', color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(df.index, df['LOD_Anomaly'], color=color1, linewidth=2, label='LOD Anomaly')
        ax1.tick_params(axis='y', labelcolor=color1)

        # Secondary Axis: Regional Mass Anomaly
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel(f'{region_name} Mass Anomaly (LWE cm)', color=color2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(df.index, df['Mass_Anomaly'], color=color2, linewidth=2, linestyle='--', label='Mass Anomaly')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Title & Combined Legend
        plt.title(
            f'TURA-GeoSync: Earth Rotation (LOD) vs {region_name} Mass Transfer\nPearson Correlation r = {r_coeff:.2f}',
            fontsize=14,
            fontweight='bold'
        )
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        fig.tight_layout()
        plt.show()

    return r_coeff, p_val


if __name__ == "__main__":
    grace_da = process_grace_lithosphere()
    lod_df = fetch_lod_from_paris_obs()

    if grace_da is not None and lod_df is not None:
        synced_lod = synchronize_lod_to_grace(lod_df, grace_da)

        himalaya_cfg = REGIONS["himalayas"]
        himalaya_mass = extract_regional_mass_anomaly(
            grace_da,
            lat_min=himalaya_cfg["lat_min"],
            lat_max=himalaya_cfg["lat_max"],
            lon_min=himalaya_cfg["lon_min"],
            lon_max=himalaya_cfg["lon_max"]
        )

        plot_correlation(synced_lod, himalaya_mass, himalaya_cfg["name"])