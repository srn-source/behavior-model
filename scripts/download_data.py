"""Download the pinned Twin-2K-500 snapshot into data/raw/twin2k500 (~680 MB).

Run: python -m scripts.download_data
"""
from huggingface_hub import snapshot_download

from src import data


def main():
    path = snapshot_download(
        data.DATASET_ID,
        repo_type="dataset",
        revision=data.DATASET_REVISION,
        local_dir=data.RAW_DIR,
        allow_patterns=["*.parquet", "*.csv", "*.json", "*.md", "*.py", "*.docx", "*.xlsx"],
    )
    print(f"downloaded {data.DATASET_ID}@{data.DATASET_REVISION[:8]} to {path}")


if __name__ == "__main__":
    main()
