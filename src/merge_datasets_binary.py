import pandas as pd
from pathlib import Path
import numpy as np

# Config
ARCHIVE_DIR = Path(r"C:\Users\ashok\Downloads\archive")
DATASETS_DIR = Path(r"D:\HTTP flood attack\datasets")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Normalization maps
LABEL_MAP_2018 = {
    "benign": "Benign",
    "dos attacks-hulk": "DoS attacks-Hulk",
    "ddos attack-hoic": "DDOS attack-HOIC"
}

LABEL_MAP_2017 = {
    "benign": "Benign",
    "dos hulk": "Attack",
    "dos goldeneye": "Attack",
    "dos slowloris": "Attack",
    "dos slowhttptest": "Attack",
    "ddos": "Attack"
}

def clean_and_normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def normalize_label(label, mapping):
    norm = " ".join(str(label).strip().split()).lower()
    return mapping.get(norm, None)

def build_training_dataset():
    print("--- Building Binary Training Dataset (CSE-CIC-IDS2018) ---")
    files = ["02-15-2018.csv", "02-16-2018.csv", "02-21-2018.csv"]
    dfs = []
    
    for file_name in files:
        file_path = ARCHIVE_DIR / file_name
        print(f"Reading {file_name}...")
        df = pd.read_csv(file_path)
        df = clean_and_normalize_columns(df)
        
        # Normalize Label column to target binary subset
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
    combined_df = combined_df[combined_df["Timestamp"].notna()].copy()
    combined_df = combined_df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    
    # Sample balanced set
    benign_df = combined_df[combined_df["Label"] == "Benign"].copy()
    hulk_df = combined_df[combined_df["Label"] == "DoS attacks-Hulk"].copy()
    hoic_df = combined_df[combined_df["Label"] == "DDOS attack-HOIC"].copy()
    
    print(f"Unique Benign available: {len(benign_df)}")
    print(f"Unique Hulk available: {len(hulk_df)}")
    print(f"Unique HOIC available: {len(hoic_df)}")
    
    # Sample without replacement
    sampled_benign = benign_df.sample(n=150000, random_state=42, replace=False)
    sampled_hulk = hulk_df.sample(n=75000, random_state=42, replace=False)
    sampled_hoic = hoic_df.sample(n=75000, random_state=42, replace=False)
    
    # Map raw attack classes to binary "Attack"
    sampled_hulk["Label"] = "Attack"
    sampled_hoic["Label"] = "Attack"
    
    final_train_df = pd.concat([sampled_benign, sampled_hulk, sampled_hoic], ignore_index=True)
    
    # Re-sort temporally to preserve the chronology
    final_train_df = final_train_df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    
    # Convert Timestamp back to string format
    final_train_df["Timestamp"] = final_train_df["Timestamp"].dt.strftime("%d/%m/%Y %H:%M:%S")
    
    out_path = DATASETS_DIR / "training_binary.csv"
    final_train_df.to_csv(out_path, index=False)
    print(f"Training dataset written to {out_path} with shape {final_train_df.shape}")
    print(final_train_df["Label"].value_counts())

def build_external_dataset():
    print("--- Building Binary External Dataset (CICIDS2017) ---")
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
    
    out_path = DATASETS_DIR / "DDos_pcap_binary_external.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"External dataset written to {out_path} with shape {combined_df.shape}")
    print(combined_df["Label"].value_counts())

def main():
    build_training_dataset()
    print("\n")
    build_external_dataset()
    print("\nBinary dataset creation completed successfully!")

if __name__ == "__main__":
    main()
