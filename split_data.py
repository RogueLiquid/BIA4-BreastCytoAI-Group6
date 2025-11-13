# this file is used for correct data split, it will save the new file in BreaKHis_400X_patient_split, it should only be run once

import os, glob, shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def collect_breakhis_400x(data_dir):
    paths = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    rows = []
    print(f"Scanning {len(paths)} total files...")
    
    for p in paths:
        lp = p.lower()
        if not lp.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        if "400x" not in lp:
            continue
        
        if "benign" in lp:
            label = "benign"
        elif "malignant" in lp:
            label = "malignant"
        else:
            continue
        
        fname = os.path.basename(p)
        stem = os.path.splitext(fname)[0]
        parts = stem.split('-')
        
        # Expect: <prefix>-<year>-<pid>-<mag>-<seq>
        if len(parts) < 5:
            print(f"Warning: Skipping file with unexpected format: {fname}")
            continue
        
        prefix, year, pid, mag, seq = parts[:5]
        # Patient/slide ID: 14-4659 (what papers use)
        patient_id = f"{year}-{pid}"
        # Or you could use: patient_id = f"{prefix}-{year}-{pid}"

        rows.append({"path": p, "label": label, "patient_id": patient_id})
    
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Found {len(df)} images belonging to {df['patient_id'].nunique()} unique patients.")
    return df

def split_by_patient(df, test_size=0.2, random_state=42):
    # One label per patient (in BreaKHis, all images from one patient share the same class)
    patient_labels = df.groupby("patient_id")["label"].first().reset_index()
    
    patients = patient_labels["patient_id"].values
    y = patient_labels["label"].values
    
    # Stratified on label, but grouping by patient
    train_pat, test_pat = train_test_split(
        patients,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    train_df = df[df["patient_id"].isin(train_pat)].copy()
    test_df  = df[df["patient_id"].isin(test_pat)].copy()
    
    print(f"Train images: {len(train_df)}, patients: {train_df['patient_id'].nunique()}")
    print(f"Test images:  {len(test_df)}, patients: {test_df['patient_id'].nunique()}")
    
    return train_df, test_df

def save_split_structure(train_df, test_df, out_root):
    for split_name, df in [("train", train_df), ("test", test_df)]:
        for _, row in df.iterrows():
            label = row["label"]          # "benign" or "malignant"
            src   = row["path"]
            dst_dir = os.path.join(out_root, split_name, label)
            os.makedirs(dst_dir, exist_ok=True)
            
            dst_path = os.path.join(dst_dir, os.path.basename(src))
            
            # Copy the file; use copy2 to preserve metadata (optional)
            shutil.copy2(src, dst_path)

    print(f"Saved new structure under: {out_root}")


def extract_type_from_path(path: str) -> str | None:
    """
    Given a full image path of a BreaKHis file, 
    parse and return the <TYPE> field (e.g. 'TA', 'DCIS', etc.).
    
    Returns None if the pattern is unexpected.
    """
    fname = os.path.basename(path)
    stem = os.path.splitext(fname)[0]       # e.g. 'SOB_B_TA-14-4659-40-001'
    
    # prefix is '<BIOPSY>_<CLASS>_<TYPE>'
    prefix = stem.split('-')[0]             # 'SOB_B_TA'
    parts = prefix.split('_')               # ['SOB', 'B', 'TA']
    
    if len(parts) >= 3:
        return parts[-1]                    # 'TA'  (this is <TYPE>)
    else:
        return None
    
def report_type_distribution(df: pd.DataFrame, dataset_name: str = "dataset"):
    """
    Add a 'type' column (the <TYPE> field from filename) to df and
    print a summary table of counts per label and type.

    Returns the summary DataFrame.
    """
    df = df.copy()
    df["type"] = df["path"].apply(extract_type_from_path)
    
    summary = (
        df
        .groupby(["label", "type"])
        .size()
        .reset_index(name="count")
        .sort_values(["label", "type"])
        .reset_index(drop=True)
    )
    
    print(f"\n=== <TYPE> distribution for {dataset_name} ===")
    print(summary.to_string(index=False))
    return summary

original_root = "./BreaKHis 400X"   # where benign/malignant folders live
out_root      = "./BreaKHis_400X_patient_split"

# the code below is for data split
df = collect_breakhis_400x(original_root)
train_df, test_df = split_by_patient(df, test_size=0.2, random_state=42)
save_split_structure(train_df, test_df, out_root)

# the code below is for <type> analysis
train_type_summary = report_type_distribution(train_df, dataset_name="train")
test_type_summary  = report_type_distribution(test_df,  dataset_name="test")
