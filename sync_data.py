"""
Data Temporal Synchronization Module
Aligns high-frequency daily IERS LOD (Length of Day) time series with lower-frequency monthly GRACE observations.
"""

from typing import Optional
import pandas as pd
import xarray as xr
from iers_lod import fetch_lod_from_paris_obs
from grace_litosfer import process_grace_lithosphere


def synchronize_lod_to_grace(lod_daily_df: pd.DataFrame, grace_monthly_da: xr.DataArray) -> pd.DataFrame:
    """
    Aligns daily Length of Day (LOD) time series to match GRACE monthly observation timestamps.

    Parameters
    ----------
    lod_daily_df : pd.DataFrame
        Daily LOD DataFrame indexed by Date.
    grace_monthly_da : xr.DataArray
        Monthly GRACE xarray DataArray containing a 'time' dimension.

    Returns
    -------
    pd.DataFrame
        Synchronized LOD DataFrame matching GRACE timestamps.
    """
    print("\n[Synchronization] Starting temporal alignment phase...")

    # Extract exact observation timestamps from GRACE data
    grace_times = grace_monthly_da['time'].values
    print(f"[Synchronization] Identified {len(grace_times)} monthly timestamp anchors in GRACE series.")

    # Reindex LOD daily series to nearest GRACE observation timestamps
    lod_synchronized = lod_daily_df.reindex(grace_times, method='nearest')

    print("[Synchronization] Alignment complete.")
    return lod_synchronized


if __name__ == "__main__":
    lod_data = fetch_lod_from_paris_obs()
    grace_data = process_grace_lithosphere()

    if lod_data is not None and grace_data is not None:
        synced_lod = synchronize_lod_to_grace(lod_data, grace_data)
        print("\n--- SYNCHRONIZED LOD DATA (MATCHED TO GRACE TIMESTAMPS) ---")
        print(synced_lod.head())