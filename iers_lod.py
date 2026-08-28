"""
IERS Length of Day (LOD) Data Ingestion Module
Fetches and cleans Earth Orientation Parameters (EOP) time series from Paris Observatory (IERS primary mirror).
"""

import ssl
from typing import Optional
import pandas as pd
from config import IERS_EOP_URL

# Relax SSL certificate context for environments with custom network configurations
ssl._create_default_https_context = ssl._create_unverified_context


def fetch_lod_from_paris_obs(url: str = IERS_EOP_URL) -> Optional[pd.DataFrame]:
    """
    Fetches Length of Day (LOD) time-series data from the IERS primary data source
    at Paris Observatory (hpiers).

    Parameters
    ----------
    url : str
        URL endpoint for IERS EOP C04 time series data.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame indexed by Date with columns ['LOD', 'LOD_err'], or None on failure.
    """
    print(f"[IERS LOD] Connecting to IERS primary endpoint: {url} ...")

    try:
        # Read whitespace-delimited fixed format file, skipping initial metadata lines (14 header lines)
        df = pd.read_csv(url, sep=r'\s+', header=None, skiprows=14)

        # Map column indices according to IERS EOP C04 format specification:
        # 0: Year, 1: Month, 2: Day, 3: MJD, 4: x, 5: y, 6: UT1-UTC, 7: LOD, ..., 13: LOD_err
        df.rename(columns={0: 'Year', 1: 'Month', 2: 'Day', 7: 'LOD', 13: 'LOD_err'}, inplace=True)

        # Construct datetime index
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

        # Isolate relevant LOD columns and sort chronologically
        lod_timeseries = df[['Date', 'LOD', 'LOD_err']].copy()
        lod_timeseries.set_index('Date', inplace=True)
        lod_timeseries.sort_index(inplace=True)

        print(f"[IERS LOD] Successfully retrieved {len(lod_timeseries)} daily records.")
        return lod_timeseries

    except Exception as exc:
        print(f"[IERS LOD] Error fetching or parsing data: {exc}")
        return None


# Alias for backward compatibility and clean naming
fetch_lod_data = fetch_lod_from_paris_obs


if __name__ == "__main__":
    lod_df = fetch_lod_from_paris_obs()
    if lod_df is not None:
        print("\nRecent 5 Days of LOD Data (Excess length of day in ms):")
        print(lod_df.tail())