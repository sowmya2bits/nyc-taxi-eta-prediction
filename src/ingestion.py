import os
import zipfile
import pandas as pd

def ingest_data(
    raw_dir: str = "data/raw", 
    sample_size: int | None = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Extracts train.zip inside data/raw/ and loads dataset into DataFrame."""
    csv_path = os.path.join(raw_dir, "train.csv")
    train_zip_path = os.path.join(raw_dir, "train.zip")
    main_zip_path = os.path.join(raw_dir, "nyc-taxi-trip-duration.zip")

    # Step 1: Extract CSV if it doesn't exist yet
    if not os.path.exists(csv_path):
        if os.path.exists(train_zip_path):
            print("[Ingestion] Extracting train.zip archive...")
            with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
        elif os.path.exists(main_zip_path):
            print("[Ingestion] Extracting nyc-taxi-trip-duration.zip archive...")
            with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_dir)
            # Handle nested train.zip if present inside main archive
            if os.path.exists(train_zip_path) and not os.path.exists(csv_path):
                with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_dir)
        else:
            raise FileNotFoundError(
                f"Could not find 'train.csv' or 'train.zip' inside '{raw_dir}'."
            )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Failed to extract train.csv into {raw_dir}")

    # Step 2: Read raw CSV and log basic metadata
    print("[Ingestion] Reading raw CSV dataset...")
    df = pd.read_csv(csv_path)
    
    total_rows, total_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"[Ingestion] Full dataset loaded: {total_rows:,} rows | {total_cols} columns | {memory_mb:.2f} MB")

    # Step 3: Optional sampling for fast development
    if sample_size and total_rows > sample_size:
        print(f"[Ingestion] Sampling {sample_size:,} records for local development...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = ingest_data()