"""
NASA EarthData GRACE Dataset Downloader Module
Automates downloading of GRACE/GRACE-FO mascon products via CMR API using earthaccess.
"""

from pathlib import Path
from typing import Union
import earthaccess
from config import DATA_DIR


def download_grace_with_earthaccess(output_dir: Union[str, Path] = DATA_DIR) -> None:
    """
    Authenticates with NASA EarthData CMR system and downloads the JPL GRACE/GRACE-FO Mascon dataset.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Target directory to save downloaded NetCDF files. Defaults to config.DATA_DIR.
    """
    target_path = Path(output_dir)
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    print("[NASA Downloader] Connecting to NASA EarthData cloud system...")

    # Authenticate (uses interactive login on first run and persists credentials to .netrc)
    auth = earthaccess.login(persist=True)

    if not auth:
        print("[NASA Downloader] Authentication failed. Please verify your EarthData credentials.")
        return

    print("[NASA Downloader] Authentication successful! Querying CMR API for GRACE mascon products...")

    # Search for JPL GRACE RL06.3 V4 Mascon dataset granules
    results = earthaccess.search_data(
        short_name="TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4",
        count=2000
    )

    if not results:
        print("[NASA Downloader] No granules found. Product short_name may have updated.")
        return

    print(f"[NASA Downloader] Found {len(results)} granules. Starting download to: {target_path}...")

    # Download granules with automatic retry, thread management, and S3 redirection
    earthaccess.download(
        results,
        local_path=str(target_path)
    )

    print("\n[NASA Downloader] All dataset granules successfully downloaded!")


if __name__ == "__main__":
    download_grace_with_earthaccess()