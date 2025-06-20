# scripts/create_sample_data.py
import pandas as pd
import numpy as np
import os


def create_sample_data():
    # Create directories if needed
    os.makedirs("data/raw", exist_ok=True)

    # All 36 Nigerian states + FCT
    nigerian_states = [
        "Abia",
        "Adamawa",
        "Akwa Ibom",
        "Anambra",
        "Bauchi",
        "Bayelsa",
        "Benue",
        "Borno",
        "Cross River",
        "Delta",
        "Ebonyi",
        "Edo",
        "Ekiti",
        "Enugu",
        "Gombe",
        "Imo",
        "Jigawa",
        "Kaduna",
        "Kano",
        "Katsina",
        "Kebbi",
        "Kogi",
        "Kwara",
        "Lagos",
        "Nasarawa",
        "Niger",
        "Ogun",
        "Ondo",
        "Osun",
        "Oyo",
        "Plateau",
        "Rivers",
        "Sokoto",
        "Taraba",
        "Yobe",
        "Zamfara",
        "FCT Abuja",
    ]

    # Generate synthetic data (2015-2024)
    np.random.seed(42)
    years = list(range(2015, 2025))
    data = []

    for year in years:
        for state in nigerian_states:
            # Create realistic correlations
            rainfall = np.random.normal(1800, 400)
            urbanization = np.random.uniform(15, 85)
            drainage = max(0.3, min(0.9, 0.7 - (urbanization / 200)))

            # Flood occurrence probability based on factors
            base_prob = 0.2 + (rainfall - 1600) / 1000 + (100 - drainage * 100) / 200
            flood_prob = max(0.05, min(0.95, base_prob))  # Clamped between 5%-95%

            data.append(
                {
                    "State": state,
                    "Year": year,
                    "Rainfall_mm": rainfall,
                    "River_Level_m": np.random.uniform(2.5, 8.0),
                    "Drainage_Efficiency": drainage,
                    "Urbanization_Rate": urbanization,
                    "Flood_Occurrence": np.random.choice(
                        [0, 1], p=[1 - flood_prob, flood_prob]
                    ),
                }
            )

    df = pd.DataFrame(data)
    df.to_csv("data/raw/nigeria_flood_data.csv", index=False)
    print(
        f"✅ Created sample data with {len(df)} records at data/raw/nigeria_flood_data.csv"
    )
    print(f"States included: {', '.join(nigerian_states)}")


if __name__ == "__main__":
    create_sample_data()
