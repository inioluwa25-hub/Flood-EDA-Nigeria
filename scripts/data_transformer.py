# scripts/data_transformer.py
import pandas as pd
from scipy import stats
import os
import boto3
from dotenv import load_dotenv

# Attempt to import boto3 with fallback
try:
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("⚠️ boto3 not available - S3 upload disabled")

load_dotenv()


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
    processed_path = "data/processed/transformed_flood_data.csv"
    raw_df.to_csv(processed_path, index=False)
    print(f"✅ Transformed data saved to {processed_path}")

    # 7. Upload to S3
    def upload_to_s3():
        if not AWS_AVAILABLE:
            print("⏩ Skipping S3 upload (boto3 not available)")
            return

        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                region_name="us-east-1",
            )

            # Verify bucket exists
            s3.head_bucket(Bucket="nigeria-flood-model-data-eda")

            s3.upload_file(
                processed_path,
                "nigeria-flood-model-data-eda",
                "processed/transformed_flood_data.csv",
                ExtraArgs={"ACL": "bucket-owner-full-control"},
            )
            print("✅ Uploaded processed data to S3")
        except Exception as e:
            print(f"❌ S3 Upload Error: {str(e)}")

    upload_to_s3()
    return raw_df


if __name__ == "__main__":
    transform_data()
