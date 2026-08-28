"""
GRACE Lithosphere Mass Anomaly Processing Module
Opens NASA GRACE NetCDF mascon data, filters out polar ice sheet melt signals (Greenland & Antarctica)
and ocean noise, isolating continental lithospheric mass variations.
"""

from pathlib import Path
from typing import Union, Optional
import xarray as xr
from config import DEFAULT_GRACE_NC_PATH


def process_grace_lithosphere(nc_file_path: Union[str, Path] = DEFAULT_GRACE_NC_PATH) -> Optional[xr.DataArray]:
    """
    Loads and filters GRACE NetCDF dataset for lithospheric mass anomaly analysis.

    Parameters
    ----------
    nc_file_path : Union[str, Path]
        Path to GRACE NetCDF file (.nc). Defaults to path defined in config.py.

    Returns
    -------
    Optional[xr.DataArray]
        Filtered xarray DataArray of Liquid Water Equivalent (lwe_thickness) mass anomalies,
        or None if file loading fails.
    """
    nc_path = Path(nc_file_path)
    print(f"[GRACE Lithosphere] Loading GRACE mascon dataset (Lazy Chunking): {nc_path}")

    if not nc_path.exists():
        print(f"[GRACE Lithosphere] File not found: {nc_path}")
        return None

    try:
        # Load dataset lazily with xarray auto-chunking to optimize memory consumption
        ds = xr.open_dataset(nc_path, chunks='auto')

        # Extract Liquid Water Equivalent thickness mass anomaly variable
        if 'lwe_thickness' in ds:
            mass_anomaly = ds['lwe_thickness']
        else:
            # Fallback for alternative variable naming conventions
            var_name = list(ds.data_vars.keys())[0]
            mass_anomaly = ds[var_name]

        # 1. Mask Polar Ice Sheets (Antarctica & Greenland melt signals)
        print("[GRACE Lithosphere] Isolating solid lithosphere signals by masking Antarctic & Greenland ice sheet melt...")
        mask_antarctica = mass_anomaly.lat < -60
        mask_greenland = (mass_anomaly.lat > 58) & (mass_anomaly.lon > 285) & (mass_anomaly.lon < 350)

        # Apply ice-sheet mask using lazy evaluation (~ is bitwise NOT)
        filtered_mass = mass_anomaly.where(~mask_antarctica).where(~mask_greenland)

        # 2. Mask Ocean Pixels
        if 'land_mask' in ds.variables:
            print("[GRACE Lithosphere] Internal land mask found. Filtering ocean grids...")
            filtered_mass = filtered_mass.where(ds['land_mask'] == 1)
        else:
            print("[GRACE Lithosphere] Warning: Internal 'land_mask' variable not present. Retaining global grid.")

        print("[GRACE Lithosphere] Filtering completed successfully.")
        return filtered_mass

    except Exception as exc:
        print(f"[GRACE Lithosphere] Error processing NetCDF dataset: {exc}")
        return None


if __name__ == "__main__":
    lithosphere_data = process_grace_lithosphere()

    if lithosphere_data is not None:
        print("\n--- FILTERED LITHOSPHERE MASS ANOMALY (XARRAY DATAARRAY) ---")
        print(lithosphere_data)