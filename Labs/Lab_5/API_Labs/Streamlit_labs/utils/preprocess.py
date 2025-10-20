import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path: str, output_path: str):
    """
    Preprocess the raw salary dataset and save a cleaned CSV for model training.

    Steps:
    1. Load CSV
    2. Drop irrelevant columns
    3. Encode categorical features
    4. Handle missing values
    5. Save cleaned dataset
    """

    # Load dataset
    df = pd.read_csv(input_path)

    # Drop columns not needed for training
    # (salary and salary_currency are redundant since we use salary_in_usd)
    df = df.drop(columns=["salary", "salary_currency"], errors="ignore")

    # Handle missing values (if any)
    df = df.dropna(subset=["salary_in_usd"])  # target variable
    df = df.fillna("Unknown")

    # Categorical columns to encode
    categorical_cols = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "company_location",
        "company_size"
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Optional: reorder columns
    cols = [c for c in df.columns if c != "salary_in_usd"] + ["salary_in_usd"]
    df = df[cols]

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

    return df, encoders


if __name__ == "__main__":
    input_path = "data/salaries.csv"
    output_path = "data/cleaned_data.csv"
    preprocess_data(input_path, output_path)
