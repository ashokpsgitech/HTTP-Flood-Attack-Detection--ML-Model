import pandas as pd
from pathlib import Path
import numpy as np

# Config
ARCHIVE_DIR = Path(r"C:\Users\ashok\Downloads\archive")
DATASETS_DIR = Path(r"D:\HTTP flood attack\datasets")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    "Benign": 0,
    "DDOS attack-HOIC": 1,
    "DoS attacks-Hulk": 2,
    "DoS attacks-GoldenEye": 3,
    "DoS attacks-SlowHTTPTest": 4,
    "DoS attacks-Slowloris": 5
}

# Normalization maps
LABEL_MAP_2018 = {
    "benign": "Benign",
    "dos attacks-goldeneye": "DoS attacks-GoldenEye",
    "dos attacks-slowloris": "DoS attacks-Slowloris",
    "dos attacks-hulk": "DoS attacks-Hulk",
    "dos attacks-slowhttptest": "DoS attacks-SlowHTTPTest",
    "ddos attack-hoic": "DDOS attack-HOIC"
}

LABEL_MAP_2017 = {
    "benign": "Benign",
    "dos hulk": "DoS attacks-Hulk",
    "dos goldeneye": "DoS attacks-GoldenEye",
    "dos slowloris": "DoS attacks-Slowloris",
    "dos slowhttptest": "DoS attacks-SlowHTTPTest",
    "ddos": "DDOS attack-HOIC"
}

def normalize_label(label, mapping):
    norm = " ".join(str(label).strip().split()).lower()
    return mapping.get(norm, None)

def clean_and_normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def build_training_dataset():
    print("--- Building Training Dataset (CSE-CIC-IDS2018) ---")
    files = ["02-15-2018.csv", "02-16-2018.csv", "02-21-2018.csv"]
    dfs = []
    
    for file_name in files:
        file_path = ARCHIVE_DIR / file_name
        print(f"Reading {file_name}...")
        df = pd.read_csv(file_path)
        df = clean_and_normalize_columns(df)
        
        # Normalize Label column
        df["Label"] = df["Label"].map(lambda x: normalize_label(x, LABEL_MAP_2018))
        df = df[df["Label"].notna()].copy()
        
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape before deduplication: {combined_df.shape}")
    
    # Deduplicate
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    print(f"Combined shape after deduplication: {combined_df.shape}")
    
    # Parse timestamps for sorting
    print("Parsing timestamps...")
    combined_df["Timestamp"] = pd.to_datetime(
        combined_df["Timestamp"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )
    # Remove any invalid timestamps
    combined_df = combined_df[combined_df["Timestamp"].notna()].copy()
    
    # Sort chronologically
    combined_df = combined_df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    
    # Class balancing (sample 50k per class)
    balanced_dfs = []
    for label in TARGET_CLASSES.keys():
        class_df = combined_df[combined_df["Label"] == label].copy()
        n_samples = len(class_df)
        print(f"Class '{label}': {n_samples} unique samples found.")
        
        if n_samples >= 50000:
            sampled_df = class_df.sample(n=50000, random_state=42, replace=False)
        else:
            print(f"Oversampling '{label}' (sampling with replacement) to reach 50,000 rows.")
            sampled_df = class_df.sample(n=50000, random_state=42, replace=True)
            
        balanced_dfs.append(sampled_df)
        
    final_train_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Re-sort temporally to preserve the chronology after balancing
    final_train_df = final_train_df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    
    # Convert Timestamp back to the string format used by run_pipeline.py
    final_train_df["Timestamp"] = final_train_df["Timestamp"].dt.strftime("%d/%m/%Y %H:%M:%S")
    
    out_path = DATASETS_DIR / "training_balanced_new.csv"
    final_train_df.to_csv(out_path, index=False)
    print(f"Training dataset written to {out_path} with shape {final_train_df.shape}")
    print(final_train_df["Label"].value_counts())

def build_external_dataset():
    print("--- Building External Dataset (CICIDS2017) ---")
    pcap_files = [
        "Wednesday-workingHours.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    ]
    
    dfs = []
    
    for file_name in pcap_files:
        file_path = ARCHIVE_DIR / file_name
        print(f"Reading {file_name}...")
        df = pd.read_csv(file_path)
        df = clean_and_normalize_columns(df)
        
        # Normalize Label column
        df["Label"] = df["Label"].map(lambda x: normalize_label(x, LABEL_MAP_2017))
        df = df[df["Label"].notna()].copy()
        
        dfs.append(df)
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined external shape: {combined_df.shape}")
    
    # Format labels column
    out_path = DATASETS_DIR / "DDos_pcap_external.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"External dataset written to {out_path} with shape {combined_df.shape}")
    print(combined_df["Label"].value_counts())

def main():
    build_training_dataset()
    print("\n")
    build_external_dataset()
    print("\nDataset creation completed successfully!")

if __name__ == "__main__":
    main()
