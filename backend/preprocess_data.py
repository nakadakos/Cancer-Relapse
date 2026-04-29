import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Target columns for the unified dataset
TARGET_COLUMNS = [
    'Age', 'Sex', 'Athleticity', 'BMI', 'Smoking_Alcohol_History',
    'Cancer_Type', 'Tumor_Stage', 'Tumor_Grade', 'Tumor_Size_cm',
    'Lymph_Nodes_Involved', 'Metastasis', 'Tumor_Type',
    'Hormone_Receptor', 'Gene_Mutations', 'Surgery_Type',
    'Chemotherapy', 'Radiation_Therapy', 'Hormone_Therapy',
    'Immunotherapy', 'Time_Since_Treatment_Months', 'Follow_Up_Visits',
    'Previous_Reoccurrence', 'Relapse'
]

np.random.seed(29)


def age_range_to_numeric(age_str):
    if pd.isna(age_str):
        return 50
    parts = str(age_str).split('-')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) // 2
    return 50


def tumor_size_range_to_numeric(size_str):
    if pd.isna(size_str):
        return 2.5
    parts = str(size_str).split('-')
    if len(parts) == 2:
        return round((int(parts[0]) + int(parts[1])) / 2 / 10, 1)  # mm to cm
    return 2.5


def inv_nodes_to_binary(nodes_str):
    if pd.isna(nodes_str):
        return 'No'
    parts = str(nodes_str).split('-')
    if len(parts) == 2 and int(parts[0]) > 0:
        return 'Yes'
    return 'No'


def process_uci_breast_cancer():
    filepath = os.path.join(RAW_DIR, 'uci_breast_cancer.csv')
    if not os.path.exists(filepath):
        print("  [SKIP] UCI Breast Cancer not found")
        return pd.DataFrame()

    print("  Processing UCI Breast Cancer (Ljubl)...")
    df = pd.read_csv(filepath)

    records = []
    for _, row in df.iterrows():
        # deg_malig is "degree of malignancy" (1-3)
        # Tumor_Stage is imputed from a realistic distribution for early-stage
        # breast cancer (Ljubl was mostly stage I-II).
        deg = min(max(int(row.get('deg_malig', 2)), 1), 3)
        record = {
            'Age': age_range_to_numeric(row.get('age')),
            'Sex': 'Female',
            'Athleticity': np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2]),
            'BMI': round(np.random.normal(27, 5), 1),
            'Smoking_Alcohol_History': np.random.choice(['None', 'Occasional', 'Frequent', 'Heavy'], p=[0.5, 0.3, 0.15, 0.05]),
            'Cancer_Type': 'Breast',
            'Tumor_Stage': np.random.choice([1, 2, 3, 4], p=[0.35, 0.45, 0.15, 0.05]),
            'Tumor_Grade': deg,
            'Tumor_Size_cm': tumor_size_range_to_numeric(row.get('tumor_size')),
            'Lymph_Nodes_Involved': inv_nodes_to_binary(row.get('inv_nodes')),
            'Metastasis': 'No',
            'Tumor_Type': 'Malignant',
            'Hormone_Receptor': np.random.choice(['Positive', 'Negative'], p=[0.7, 0.3]),
            'Gene_Mutations': np.random.choice(['None', 'TP53', 'BRCA1/2', 'Other'], p=[0.6, 0.2, 0.1, 0.1]),
            'Surgery_Type': np.random.choice(['Lumpectomy', 'Mastectomy', 'None'], p=[0.4, 0.4, 0.2]),
            'Chemotherapy': np.random.choice(['Yes', 'No'], p=[0.6, 0.4]),
            'Radiation_Therapy': 'Yes' if str(row.get('irradiat', 'no')).strip().lower() == 'yes' else 'No',
            'Hormone_Therapy': np.random.choice(['Yes', 'No'], p=[0.5, 0.5]),
            'Immunotherapy': np.random.choice(['Yes', 'No'], p=[0.2, 0.8]),
            'Time_Since_Treatment_Months': np.random.randint(6, 72),
            'Follow_Up_Visits': np.random.randint(2, 20),
            'Previous_Reoccurrence': 'No',
            'Relapse': 'Yes' if str(row.get('class', '')).strip() == 'recurrence-events' else 'No',
        }
        records.append(record)

    result = pd.DataFrame(records)
    print(f"  [OK] {len(result)} records. Relapse: {result['Relapse'].value_counts().to_dict()}")
    return result


def process_wpbc():
    """Process Wisconsin Prognostic Breast Cancer dataset."""
    filepath = os.path.join(RAW_DIR, 'wpbc.csv')
    if not os.path.exists(filepath):
        print("  [SKIP] WPBC not found")
        return pd.DataFrame()

    print("  Processing WPBC...")
    df = pd.read_csv(filepath)

    records = []
    for _, row in df.iterrows():
        tumor_size = row.get('tumor_size', np.nan)
        if pd.isna(tumor_size):
            tumor_size = round(np.random.gamma(2.0, 2.0), 1)
        
        lymph = row.get('lymph_node_status', np.nan)
        lymph_involved = 'Yes' if (not pd.isna(lymph) and float(lymph) > 0) else 'No'
        
        worst_radius = row.get('worst_radius', np.nan)
        grade = 1
        if not pd.isna(worst_radius):
            if worst_radius > 20:
                grade = 3
            elif worst_radius > 15:
                grade = 2

        record = {
            # WPBC does not include patient age; it is imputed from U(30,79).
            # This feature adds noise for these 198 rows. but it keeps the dataset consistent.
            'Age': np.random.randint(30, 80),
            'Sex': 'Female',
            'Athleticity': np.random.choice(['Low', 'Medium', 'High'], p=[0.4, 0.4, 0.2]),
            'BMI': round(np.random.normal(27, 5), 1),
            'Smoking_Alcohol_History': np.random.choice(['None', 'Occasional', 'Frequent', 'Heavy'], p=[0.5, 0.3, 0.15, 0.05]),
            'Cancer_Type': 'Breast',
            'Tumor_Stage': np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10]),
            'Tumor_Grade': grade,
            'Tumor_Size_cm': round(float(tumor_size), 1) if not pd.isna(tumor_size) else 2.5,
            'Lymph_Nodes_Involved': lymph_involved,
            'Metastasis': 'No',
            'Tumor_Type': 'Malignant',
            'Hormone_Receptor': np.random.choice(['Positive', 'Negative'], p=[0.7, 0.3]),
            'Gene_Mutations': np.random.choice(['None', 'TP53', 'BRCA1/2', 'Other'], p=[0.6, 0.2, 0.1, 0.1]),
            'Surgery_Type': np.random.choice(['Lumpectomy', 'Mastectomy'], p=[0.5, 0.5]),
            'Chemotherapy': np.random.choice(['Yes', 'No'], p=[0.65, 0.35]),
            'Radiation_Therapy': np.random.choice(['Yes', 'No'], p=[0.55, 0.45]),
            'Hormone_Therapy': np.random.choice(['Yes', 'No'], p=[0.5, 0.5]),
            'Immunotherapy': np.random.choice(['Yes', 'No'], p=[0.2, 0.8]),
            'Time_Since_Treatment_Months': int(row.get('time', 24)),
            'Follow_Up_Visits': np.random.randint(2, 20),
            'Previous_Reoccurrence': 'No',
            'Relapse': 'Yes' if str(row.get('outcome', 'N')).strip() == 'R' else 'No',
        }
        records.append(record)

    result = pd.DataFrame(records)
    print(f"  [OK] {len(result)} records. Relapse: {result['Relapse'].value_counts().to_dict()}")
    return result


def process_thyroid_cancer():
    filepath = os.path.join(RAW_DIR, 'thyroid_cancer.csv')
    if not os.path.exists(filepath):
        print("  [SKIP] Thyroid Cancer not found")
        return pd.DataFrame()

    print("  Processing Thyroid Cancer Recurrence...")
    df = pd.read_csv(filepath)

    stage_map = {'I': 1, 'II': 2, 'III': 3, 'IVA': 4, 'IVB': 4}
    
    records = []
    for _, row in df.iterrows():
        gender = str(row.get('Gender', 'F')).strip()
        sex = 'Male' if gender in ('M', 'Male') else 'Female'
        
        smoking = str(row.get('Smoking', 'No')).strip()
        if smoking == 'Yes':
            smoke_hist = np.random.choice(['Occasional', 'Frequent', 'Heavy'], p=[0.4, 0.4, 0.2])
        else:
            smoke_hist = 'None'

        stage_str = str(row.get('Stage', 'I')).strip()
        stage = stage_map.get(stage_str, 2)

        m_status = str(row.get('M', 'M0')).strip()
        metastasis = 'Yes' if m_status == 'M1' else 'No'

        n_status = str(row.get('N', 'N0')).strip()
        lymph = 'Yes' if n_status != 'N0' else 'No'

        recurred = str(row.get('Recurred', 'No')).strip()
        relapse = 'Yes' if recurred == 'Yes' else 'No'

        rad_value = row.get('Hx Radiotherapy', row.get('Hx Radiothreapy', 'No'))
        radiation = 'Yes' if str(rad_value).strip().lower() == 'yes' else 'No'

        record = {
            'Age': int(row.get('Age', 45)),
            'Sex': sex,
            'Athleticity': np.random.choice(['Low', 'Medium', 'High'], p=[0.35, 0.40, 0.25]),
            'BMI': round(np.random.normal(26, 4), 1),
            'Smoking_Alcohol_History': smoke_hist,
            'Cancer_Type': 'Thyroid',
            'Tumor_Stage': stage,
            'Tumor_Grade': np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]),
            'Tumor_Size_cm': round(np.random.gamma(2.0, 1.5), 1),
            'Lymph_Nodes_Involved': lymph,
            'Metastasis': metastasis,
            'Tumor_Type': 'Malignant',
            'Hormone_Receptor': 'Not Applicable',
            'Gene_Mutations': np.random.choice(['None', 'TP53', 'Other'], p=[0.6, 0.25, 0.15]),
            'Surgery_Type': 'Excision',
            'Chemotherapy': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
            'Radiation_Therapy': radiation,
            'Hormone_Therapy': 'No',
            'Immunotherapy': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            'Time_Since_Treatment_Months': np.random.randint(12, 96),
            'Follow_Up_Visits': np.random.randint(3, 25),
            'Previous_Reoccurrence': 'No',
            'Relapse': relapse,
        }
        records.append(record)

    result = pd.DataFrame(records)
    print(f"  [OK] {len(result)} records. Relapse: {result['Relapse'].value_counts().to_dict()}")
    return result


def generate_clinical_synthetic(cancer_type, n_samples, relapse_rate, surgery_types, sex_distribution=None):
    print(f"  Generating clinically-informed data for {cancer_type} ({n_samples} samples, {relapse_rate*100:.0f}% relapse rate)...")
    
    if sex_distribution is None:
        sex_distribution = ['Male', 'Female']
    
    records = []
    for _ in range(n_samples):
        age = int(np.clip(np.random.normal(62, 12), 25, 90))
        sex = np.random.choice(sex_distribution) if isinstance(sex_distribution, list) else sex_distribution
        stage = np.random.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15])
        grade = np.random.choice([1, 2, 3], p=[0.30, 0.45, 0.25])
        tumor_size = round(np.clip(np.random.gamma(2.5, 2.0), 0.5, 15.0), 1)
        lymph = np.random.choice(['Yes', 'No'], p=[0.35, 0.65])
        metastasis = 'Yes' if stage == 4 else np.random.choice(['Yes', 'No'], p=[0.08, 0.92])
        smoking = np.random.choice(['None', 'Occasional', 'Frequent', 'Heavy'], p=[0.35, 0.30, 0.20, 0.15])
        chemo = np.random.choice(['Yes', 'No'], p=[0.60, 0.40])
        radiation = np.random.choice(['Yes', 'No'], p=[0.45, 0.55])
        surgery = np.random.choice(surgery_types)
        time_since = np.random.randint(3, 96)
        
        # Calculate relapse probability based on clinical factors
        prob = relapse_rate
        prob += (stage - 2) * 0.08
        prob += (grade - 2) * 0.05
        prob += 0.10 if lymph == 'Yes' else 0
        prob += 0.20 if metastasis == 'Yes' else 0
        prob += 0.05 if smoking in ['Frequent', 'Heavy'] else 0
        prob -= 0.08 if chemo == 'Yes' else 0
        prob -= 0.05 if radiation == 'Yes' else 0
        prob = np.clip(prob, 0.03, 0.95)
        
        relapse = 'Yes' if np.random.random() < prob else 'No'

        record = {
            'Age': age, 'Sex': sex,
            'Athleticity': np.random.choice(['Low', 'Medium', 'High'], p=[0.45, 0.35, 0.20]),
            'BMI': round(np.clip(np.random.normal(27.5, 5), 16, 45), 1),
            'Smoking_Alcohol_History': smoking,
            'Cancer_Type': cancer_type,
            'Tumor_Stage': stage, 'Tumor_Grade': grade,
            'Tumor_Size_cm': tumor_size,
            'Lymph_Nodes_Involved': lymph, 'Metastasis': metastasis,
            'Tumor_Type': np.random.choice(['Malignant', 'Benign'], p=[0.95, 0.05]),
            'Hormone_Receptor': 'Not Applicable',
            'Gene_Mutations': np.random.choice(['None', 'TP53', 'Other'], p=[0.55, 0.30, 0.15]),
            'Surgery_Type': surgery,
            'Chemotherapy': chemo,
            'Radiation_Therapy': radiation,
            'Hormone_Therapy': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            'Immunotherapy': np.random.choice(['Yes', 'No'], p=[0.25, 0.75]),
            'Time_Since_Treatment_Months': time_since,
            'Follow_Up_Visits': int(time_since / 6) + np.random.randint(0, 5),
            'Previous_Reoccurrence': np.random.choice(['Yes', 'No'], p=[0.12, 0.88]),
            'Relapse': relapse,
        }
        records.append(record)

    result = pd.DataFrame(records)
    print(f"  [OK] {len(result)} records. Relapse: {result['Relapse'].value_counts().to_dict()}")
    return result


def main():
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    print("\n--- REAL-WORLD DATASETS ---")
    dfs = []
    
    df_breast1 = process_uci_breast_cancer()
    if len(df_breast1) > 0:
        df_breast1['Data_Source'] = 'UCI_Breast_Ljubljana'
        dfs.append(df_breast1)

    df_breast2 = process_wpbc()
    if len(df_breast2) > 0:
        df_breast2['Data_Source'] = 'WPBC'
        dfs.append(df_breast2)

    df_thyroid = process_thyroid_cancer()
    if len(df_thyroid) > 0:
        df_thyroid['Data_Source'] = 'UCI_Thyroid_Recurrence'
        dfs.append(df_thyroid)

  
    print("\n--- CLINICALLY-INFORMED SYNTHETIC DATASETS ---")

    df_lung = generate_clinical_synthetic('Lung', 350, 0.38, ['Lobectomy', 'None'])
    df_lung['Data_Source'] = 'Clinical_Synthetic_Lung'
    dfs.append(df_lung)


    df_colon = generate_clinical_synthetic('Colon', 350, 0.30, ['Resection', 'None'])
    df_colon['Data_Source'] = 'Clinical_Synthetic_Colon'
    dfs.append(df_colon)


    df_prostate = generate_clinical_synthetic('Prostate', 350, 0.25, ['Prostatectomy', 'None'], sex_distribution='Male')
    df_prostate['Data_Source'] = 'Clinical_Synthetic_Prostate'
    dfs.append(df_prostate)


    df_liver = generate_clinical_synthetic('Liver', 350, 0.55, ['Excision', 'None'])
    df_liver['Data_Source'] = 'Clinical_Synthetic_Liver'
    dfs.append(df_liver)

    df_mouth = generate_clinical_synthetic('Mouth', 350, 0.40, ['Excision', 'None'])
    df_mouth['Data_Source'] = 'Clinical_Synthetic_Mouth'
    dfs.append(df_mouth)

    # Combine all
    combined = pd.concat(dfs, ignore_index=True)
    
    # Clip BMI values
    combined['BMI'] = combined['BMI'].clip(15.0, 50.0)
    
    # Ensure correct column order
    output_columns = TARGET_COLUMNS + ['Data_Source']
    combined = combined[output_columns]

    # Save
    output_path = os.path.join(PROCESSED_DIR, 'combined_cancer_data.csv')
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nTotal records: {len(combined)}")
    print(f"\nBy Cancer Type:")
    print(combined['Cancer_Type'].value_counts().to_string())
    print(f"\nBy Data Source:")
    print(combined['Data_Source'].value_counts().to_string())
    print(f"\nRelapse Distribution:")
    print(combined['Relapse'].value_counts().to_string())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
