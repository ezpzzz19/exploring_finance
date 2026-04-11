"""
Download dataset files from Google Drive into the data/ folder.

Usage:
    uv run python download_data.py

The data folder is hosted on Google Drive. This script downloads
the entire folder in one go using gdown.

Make sure the Google Drive folder is shared as:
    "Anyone with the link" → Viewer
"""

import os
import gdown

# ── Configuration ───────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FOLDER_ID = "1zCadGVIZvnRekehDiFLPfa1ogBCuqM_V"
FOLDER_URL = f"https://drive.google.com/drive/folders/{FOLDER_ID}"


def download_data(force: bool = False) -> None:
    """Download the entire data folder from Google Drive."""
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR) and not force:
        print(f"  data/ already exists with {len(os.listdir(DATA_DIR))} files.")
        print("  Run with force=True to re-download, or delete data/ first.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print("  ↓ Downloading data folder from Google Drive ...")
    gdown.download_folder(FOLDER_URL, output=DATA_DIR, quiet=False)
    print(f"\n  ✓ Done! Files saved to {DATA_DIR}/")


if __name__ == "__main__":
    download_data(force=True)
