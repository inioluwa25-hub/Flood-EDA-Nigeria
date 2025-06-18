# scripts/data_transformer.py
import pandas as pd
from scipy import stats
import os


def transform_data():
    # 1. Load raw data
    try:
        raw_df = pd.read_csv("data/raw/nigeria_flood_data.csv")
        print(f"✅ Loaded {len(raw_df)} records from raw data")
    except FileNotFoundError:
        print("❌ Error: Raw data not found. Run create_sample_data.py first")
        return

    # 2. Handle missing values
    raw_df.fillna(
        {
            "Rainfall_mm": raw_df["Rainfall_mm"].mean(),
            "River_Level_m": raw_df["River_Level_m"].median(),
            "Drainage_Efficiency": raw_df["Drainage_Efficiency"].mean(),
            "Urbanization_Rate": raw_df["Urbanization_Rate"].median(),
        },
        inplace=True,
    )

    # 3. Remove extreme outliers (3 standard deviations)
    for col in ["Rainfall_mm", "River_Level_m", "Urbanization_Rate"]:
        mean = raw_df[col].mean()
        std = raw_df[col].std()
        raw_df = raw_df[(raw_df[col] > mean - 3 * std) & (raw_df[col] < mean + 3 * std)]

    # 4. Create multimodal feature: Flood Risk Index
    raw_df["Flood_Risk_Index"] = (
        raw_df["Rainfall_mm"] * 0.4
        + (1 - raw_df["Drainage_Efficiency"]) * 0.3
        + (raw_df["Urbanization_Rate"] / 100) * 0.3
    )

    # 5. Apply Box-Cox transformation to normalize distributions
    for col in ["Rainfall_mm", "River_Level_m"]:
        # Add constant to ensure positive values
        shifted = raw_df[col] - raw_df[col].min() + 0.01
        transformed, _ = stats.boxcox(shifted)
        raw_df[f"{col}_Transformed"] = transformed

    # 6. Save processed data
    os.makedirs("data/processed", exist_ok=True)
    raw_df.to_csv("data/processed/transformed_flood_data.csv", index=False)
    print("✅ Transformed data saved to data/processed/transformed_flood_data.csv")

    return raw_df


if __name__ == "__main__":
    transform_data()
