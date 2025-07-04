# scripts/nema_data_cleaner.py
import pandas as pd
from datetime import datetime
import re
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Configuration
load_dotenv()
INPUT_FILE = "data/raw/nema-flood-data-06102022.xlsx"
OUTPUT_FILE = "data/processed/cleaned_flood_data.csv"


def clean_nema_data():
    # 1. Load raw data
    try:
        raw_df = pd.read_excel(INPUT_FILE, sheet_name="2022 NEMA Flood Data")
        print(f"✅ Loaded {len(raw_df)} records from raw data")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

    # 2. Standardize column names
    raw_df.columns = [col.strip().upper().replace(" ", "_") for col in raw_df.columns]

    # 3. Clean date column
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            if isinstance(date_str, str):
                # Handle formats like "24/08/2022"
                if "/" in date_str:
                    return datetime.strptime(date_str, "%d/%m/%Y")
                # Handle formats like "2022-08-24 00:00:00"
                else:
                    return pd.to_datetime(date_str)
            return date_str
        except:
            return pd.NaT

    raw_df["DATE"] = raw_df["DATE_OF_OCCURRENCE"].apply(parse_date)

    # 4. Clean numeric columns
    def extract_numeric(value):
        if pd.isna(value) or value == "-":
            return 0
        if isinstance(value, str):
            # Extract first numeric part from strings like "8418 persons"
            match = re.search(r"(\d+)", str(value))
            return int(match.group(1)) if match else 0
        return int(value)

    for col in ["PERSONS_AFFECTED", "DISPLACED_PERSONS"]:
        raw_df[col] = raw_df[col].apply(extract_numeric)

    # 5. Handle missing values
    raw_df["COMMUNITY"] = raw_df["COMMUNITY"].fillna("Unspecified")
    raw_df["LGA"] = raw_df["LGA"].fillna("Unspecified")

    # 6. Create derived features
    raw_df["YEAR"] = raw_df["DATE"].dt.year
    raw_df["MONTH"] = raw_df["DATE"].dt.month
    raw_df["SEASON"] = raw_df["MONTH"].apply(
        lambda m: "Dry" if m in [11, 12, 1, 2] else "Rainy"
    )

    # Calculate severity ratio (displaced/persons affected)
    raw_df["SEVERITY_RATIO"] = raw_df.apply(
        lambda x: (
            x["DISPLACED_PERSONS"] / x["PERSONS_AFFECTED"]
            if x["PERSONS_AFFECTED"] > 0
            else 0
        ),
        axis=1,
    )

    # 7. Remove duplicates
    raw_df = raw_df.drop_duplicates()

    # 8. Save cleaned data
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    raw_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Cleaned data saved to {OUTPUT_FILE}")

    # 9. Generate basic visualizations
    plot_distributions(raw_df)

    return raw_df


def plot_distributions(df):
    """Generate basic distribution plots for key columns"""
    plt.figure(figsize=(15, 10))

    # Persons affected distribution
    plt.subplot(2, 2, 1)
    df["PERSONS_AFFECTED"].plot(kind="hist", bins=50, logy=True)
    plt.title("Distribution of Persons Affected (Log Scale)")

    # Floods by state
    plt.subplot(2, 2, 2)
    df["STATE"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 States by Flood Frequency")

    # Monthly distribution
    plt.subplot(2, 2, 3)
    df["MONTH"].value_counts().sort_index().plot(kind="bar")
    plt.title("Floods by Month")
    plt.xticks(
        range(12),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )

    # Severity ratio
    plt.subplot(2, 2, 4)
    df["SEVERITY_RATIO"].plot(kind="box")
    plt.title("Distribution of Severity Ratio")

    plt.tight_layout()
    plt.savefig("data/processed/data_distributions.png")
    plt.close()


if __name__ == "__main__":
    cleaned_data = clean_nema_data()
