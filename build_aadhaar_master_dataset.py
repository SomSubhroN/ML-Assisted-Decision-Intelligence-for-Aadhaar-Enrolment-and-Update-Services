import pandas as pd
import os
from glob import glob

# ==================================================
# 1. LOAD & CLEAN
# ==================================================

def load_and_clean_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Robust date parsing (UIDAI-safe)
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    df = df.dropna(subset=["date"])
    return df


# ==================================================
# 2. ENROLMENT PROCESSING
# ==================================================

def process_enrolment_folder(folder):
    frames = []

    for file in glob(os.path.join(folder, "*.csv")):
        df = load_and_clean_csv(file)

        df_long = df.melt(
            id_vars=["date", "state", "district", "pincode"],
            value_vars=["age_0_5", "age_5_17", "age_18_greater"],
            var_name="age_group",
            value_name="count"
        )

        df_long["age_group"] = df_long["age_group"].map({
            "age_0_5": "0_5",
            "age_5_17": "5_17",
            "age_18_greater": "18_plus"
        })

        df_long["source_type"] = "enrolment"
        frames.append(df_long)

    return pd.concat(frames, ignore_index=True)


# ==================================================
# 3. BIOMETRIC PROCESSING
# ==================================================

def process_biometric_folder(folder):
    frames = []

    for file in glob(os.path.join(folder, "*.csv")):
        df = load_and_clean_csv(file)

        df_long = df.melt(
            id_vars=["date", "state", "district", "pincode"],
            value_vars=["bio_age_5_17", "bio_age_17_"],
            var_name="age_group",
            value_name="count"
        )

        df_long["age_group"] = df_long["age_group"].map({
            "bio_age_5_17": "5_17",
            "bio_age_17_": "18_plus"
        })

        df_long["source_type"] = "biometric"
        frames.append(df_long)

    return pd.concat(frames, ignore_index=True)


# ==================================================
# 4. DEMOGRAPHIC PROCESSING
# ==================================================

def process_demographic_folder(folder):
    frames = []

    for file in glob(os.path.join(folder, "*.csv")):
        df = load_and_clean_csv(file)

        df_long = df.melt(
            id_vars=["date", "state", "district", "pincode"],
            value_vars=["demo_age_5_17", "demo_age_17_"],
            var_name="age_group",
            value_name="count"
        )

        df_long["age_group"] = df_long["age_group"].map({
            "demo_age_5_17": "5_17",
            "demo_age_17_": "18_plus"
        })

        df_long["source_type"] = "demographic"
        frames.append(df_long)

    return pd.concat(frames, ignore_index=True)


# ==================================================
# 5. BUILD MASTER DATASET
# ==================================================

def build_master_dataset(enrol_dir, bio_dir, demo_dir):
    enrol_df = process_enrolment_folder(enrol_dir)
    bio_df = process_biometric_folder(bio_dir)
    demo_df = process_demographic_folder(demo_dir)

    master = pd.concat([enrol_df, bio_df, demo_df], ignore_index=True)

    master["count"] = master["count"].fillna(0).astype(int)

    master = master.sort_values(
        ["date", "state", "district", "pincode", "source_type", "age_group"]
    )

    return master


# ==================================================
# 6. RUN SCRIPT
# ==================================================

if __name__ == "__main__":
    ENROLMENT_DIR = r"C:\Users\User\Desktop\UIDAI Hackathon\enrolment"
    BIOMETRIC_DIR = r"C:\Users\User\Desktop\UIDAI Hackathon\biometric"
    DEMOGRAPHIC_DIR = r"C:\Users\User\Desktop\UIDAI Hackathon\demographic"
    OUTPUT_PATH = r"C:\Users\User\Desktop\UIDAI Hackathon\output\aadhaar_master_dataset.csv"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    master_df = build_master_dataset(
        ENROLMENT_DIR,
        BIOMETRIC_DIR,
        DEMOGRAPHIC_DIR
    )

    master_df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Aadhaar master dataset created successfully")
    print("Rows:", len(master_df))
    print("Columns:", list(master_df.columns))
