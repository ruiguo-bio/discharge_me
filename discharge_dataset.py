import os
import argparse
import pandas as pd
import json
import re
import logging
from tqdm import tqdm

tqdm.pandas()


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler('token_cost.log')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def load_csv(base_dir, folder, filename, **kwargs):
    path = os.path.join(base_dir, folder, filename)
    return pd.read_csv(path, **kwargs)


def load_json_file(data_folder, filename):
    path = os.path.join(data_folder, filename)
    with open(path, 'r') as f:
        return json.load(f)


def segment_input_text(input_text, patterns):
    # Initialize a dictionary to hold the segmented content
    segmented_content = {key: "" for key in patterns.keys()}
    # Precompile patterns with optional colons and case-insensitivity
    processed_patterns = {}
    for section, pattern in patterns.items():
        if isinstance(pattern, list):
            processed_patterns[section] = [
                re.compile(rf'^[\s_\-]*{re.escape(p.strip(":"))}[:]*\s*', re.IGNORECASE)
                for p in pattern
            ]
        else:
            processed_patterns[section] = [
                re.compile(rf'^[\s_\-]*{re.escape(pattern.strip(":"))}[:]*\s*', re.IGNORECASE)
            ]
    lines = input_text.split('\n')
    current_section = None
    last_matched_index = -1
    for line in lines:
        matched = False
        # Check each section’s pattern in order
        for i, (section, patterns_list) in enumerate(processed_patterns.items()):
            if matched:
                break
            for compiled_pattern in patterns_list:
                if compiled_pattern.match(line):
                    if section in ['brief hospital course', 'discharge instructions']:
                        current_section = section
                        last_matched_index = i
                        matched = True
                        break
                    else:
                        # Only update if it’s the same or a subsequent section
                        if i > last_matched_index and i < last_matched_index + 4:
                            current_section = section
                            last_matched_index = i
                            matched = True
                            break
        if current_section:
            segmented_content[current_section] += line + '\n'
    return segmented_content


def clean_text(text, pattern=r"^___ \d{2}:\d{2}(AM|PM) [A-Za-z]+\s*$"):
    # Remove lines that strictly match the pattern
    cleaned_text = re.sub(pattern, "", text, flags=re.MULTILINE)
    # Remove extra blank lines
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
    return cleaned_text.strip()


def main(args):
    logger = setup_logging()
    base_dir = args.data_dir
    dataset_dir = os.path.join(base_dir, "dataset")

    logger.info("Loading discharge CSV files...")
    read_kwargs = {
        "compression": "gzip",
        "header": 0,
        "sep": ",",
        "quotechar": '"'
    }
    df_train = load_csv(base_dir, "train", "discharge.csv.gz", **read_kwargs)
    df_valid = load_csv(base_dir, "valid", "discharge.csv.gz", **read_kwargs)
    df_test1 = load_csv(base_dir, "test_phase_1", "discharge.csv.gz", **read_kwargs)
    df_test2 = load_csv(base_dir, "test_phase_2", "discharge.csv.gz", **read_kwargs)

    logger.info("Loading JSON dictionaries...")
    patients_adm_dict = load_json_file(dataset_dir, "patients_admissions_dict.json")
    transfer_sum_dict = load_json_file(dataset_dir, "transfer_summary_dict.json")
    diagnoses_dict = load_json_file(dataset_dir, "diagnoses_dict.json")

    # Ensure hadm_id is a string for merging
    for df in [df_train, df_valid, df_test1, df_test2]:
        df['hadm_id'] = df['hadm_id'].astype(str)

    # Create DataFrames from the JSON dictionaries
    df_patients = pd.DataFrame(patients_adm_dict.items(), columns=['hadm_id', 'patients_admissions'])
    df_transfer = pd.DataFrame(transfer_sum_dict.items(), columns=['hadm_id', 'transfer_summary'])
    df_diagnoses = pd.DataFrame(diagnoses_dict.items(), columns=['hadm_id', 'diagnoses'])

    dfs = {
        "train": df_train,
        "valid": df_valid,
        "test1": df_test1,
        "test2": df_test2,
    }

    for key, df in dfs.items():
        df = df.merge(df_patients, on='hadm_id', how='left')
        df['patients_admissions'] = df['patients_admissions'].fillna('')
        df = df.merge(df_transfer, on='hadm_id', how='left')
        df['transfer_summary'] = df['transfer_summary'].fillna('')
        df = df.merge(df_diagnoses, on='hadm_id', how='left')
        df['diagnoses'] = df['diagnoses'].fillna('')
        # Concatenate additional info and drop original columns
        df['additional_info'] = df[['patients_admissions', 'transfer_summary', 'diagnoses']].apply(
            lambda x: '\n'.join([f"{k}: {v}" for k, v in x.items()]), axis=1
        )
        df.drop(columns=['patients_admissions', 'transfer_summary', 'diagnoses'], inplace=True)
        # Extract text starting from "Allergies:" and append additional_info
        df['text'] = df['text'].str.extract(r'(Allergies:.*)', flags=re.DOTALL)
        df['text'] = df[['additional_info', 'text']].apply(lambda x: '\n'.join(x), axis=1)
        # Optionally convert hadm_id to int if needed
        if df['hadm_id'].iloc[0].isdigit():
            df['hadm_id'] = df['hadm_id'].astype(int)
        # Update the dictionary entry with the modified DataFrame
        dfs[key] = df

    # Now reassign back to the original variables
    df_train = dfs["train"]
    df_valid = dfs["valid"]
    df_test1 = dfs["test1"]
    df_test2 = dfs["test2"]

    # Define segmentation patterns
    patterns = {
        "patients_admissions": "patients_admissions:",
        "transfer_summary": "transfer_summary:",
        "diagnoses": "diagnoses:",
        "Service": "Service:",
        "Allergies": "Allergies:",
        "Attending": "Attending:",
        "Chief Complaint": "Chief Complaint:",
        "Major Surgical or Invasive Procedure": "Major Surgical or Invasive Procedure:",
        "History of Present Illness": "History of Present Illness:",
        "REVIEW OF SYSTEMS:": ["REVIEW OF SYSTEMS:", "ros"],
        "Past Medical History": ["Past Medical History:", "Oncologic History", "Other Past Medical History"],
        "Social History": "Social History:",
        "Family History": "Family History:",
        "Physical Exam": [
            'physical examination', 'admission vitals', 'discharge vitals', "PHYSICAL EXAM:",
            'ADMISSION PHYSICAL EXAM', 'physical exam on admission', 'physical exam on discharge',
            'admission physical examination', 'discharge physical examination', 'discharge PHYSICAL EXAM',
            'discharge EXAM', 'admission EXAM', 'vital signs', 'exam on discharge', 'exam on admission'
        ],
        "Pertinent Results": [
            "Pertinent Results:", "labs on admission", "labs on discharge", "admission labs",
            "discharge labs", 'OTHER LABS', 'CSF Studies', "PERTINENT LABS", 'microbiology',
            'CBC w/ diff', 'Rheumatologic testing', 'ekg', 'ecg'
        ],
        "Imaging and Studies": [
            "IMAGING:", 'imaging/studies', "IMPRESSION", "STUDIES:", "CXR", "mri", "echo", "cat scan", 'cta'
        ],
        "brief hospital course": "brief hospital course:",
        "admission medications": ["admission medications:", "medications on admission"],
        "Discharge Medications": ["Discharge Medications:", "medications on discharge"],
        "Discharge Disposition": "Discharge Disposition:",
        "Discharge Diagnosis": ["Discharge Diagnosis:", "primary diagnosis", "secondary diagnosis"],
        "Discharge Condition": "Discharge Condition:",
        "Discharge Instructions": "Discharge Instructions:",
        "Followup Instructions": "Followup Instructions:",
        "Provider": "Provider:",
        "code status": "code status:",
    }

    # Ensure hadm_id is int for segmentation processing
    for df in [df_train, df_valid, df_test1, df_test2]:
        df['hadm_id'] = df['hadm_id'].astype(int)

    # Apply segmentation on the text column using progress_apply
    for df in [df_train, df_valid, df_test1, df_test2]:
        df['segmented_content'] = df['text'].progress_apply(segment_input_text, patterns=patterns)
        df['segmented_content_length'] = df['segmented_content'].apply(lambda x: {k: len(v) for k, v in x.items()})

    # Compute statistics on section lengths and cap them at the 90% quantile
    lengths_df = pd.DataFrame(df_train['segmented_content_length'].tolist()).fillna(0)
    lengths_df.describe([.25, .5, .75, .9, .95]).to_excel(os.path.join(dataset_dir, 'lengths_df_statistics.xlsx'))
    quantile_90 = lengths_df.quantile(0.90)

    for df in [df_train, df_valid, df_test1, df_test2]:
        df['segmented_content'] = df['segmented_content'].apply(
            lambda x: {k: v[:int(quantile_90.get(k, len(v)))] for k, v in x.items()}
        )

    # Expand segmented content into new columns and drop unneeded columns
    def expand_df(df):
        expanded = df['segmented_content'].apply(pd.Series)
        df_segmented = df.join(expanded).drop(columns=['segmented_content', 'text'])
        return df_segmented

    df_train_seg = expand_df(df_train)
    df_valid_seg = expand_df(df_valid)
    df_test1_seg = expand_df(df_test1)
    df_test2_seg = expand_df(df_test2)

    # Process radiology CSVs and merge them with the corresponding segmented DataFrames
    def process_radiology(file_path):
        df = pd.read_csv(file_path, compression='gzip')
        df = df.groupby('hadm_id')['text'].apply(' '.join).reset_index()
        df.rename(columns={'text': 'radiology'}, inplace=True)
        df.fillna('', inplace=True)
        return df

    rad_train = process_radiology(os.path.join(base_dir, "train", "radiology.csv.gz"))
    df_train_seg = pd.merge(df_train_seg, rad_train, on='hadm_id', how='left')

    rad_valid = process_radiology(os.path.join(base_dir, "valid", "radiology.csv.gz"))
    df_valid_seg = pd.merge(df_valid_seg, rad_valid, on='hadm_id', how='left')

    rad_test1 = process_radiology(os.path.join(base_dir, "test_phase_1", "radiology.csv.gz"))
    df_test1_seg = pd.merge(df_test1_seg, rad_test1, on='hadm_id', how='left')

    rad_test2 = process_radiology(os.path.join(base_dir, "test_phase_2", "radiology.csv.gz"))
    df_test2_seg = pd.merge(df_test2_seg, rad_test2, on='hadm_id', how='left')

    # Cap radiology texts using the maximum word count of "Imaging and Studies"
    df_train_seg['Imaging and Studies_word_count'] = df_train_seg['Imaging and Studies'].apply(lambda x: len(x.split()))
    max_word_count = df_train_seg['Imaging and Studies_word_count'].max()
    for df in [df_train_seg, df_valid_seg, df_test1_seg, df_test2_seg]:
        df['radiology'] = df['radiology'].apply(
            lambda x: x if x == '' else ' '.join(x.split()[:max_word_count])
        )
        df['Imaging and Studies'] = df.apply(
            lambda row: row['radiology'] if row['Imaging and Studies'] == '' else row['Imaging and Studies'], axis=1
        )
        df.drop(columns=['radiology'], inplace=True)

    # Rename columns as needed
    rename_map = {'brief hospital course': 'brief_hospital_course',
                  'Discharge Instructions': 'discharge_instructions'}
    df_train_seg.rename(columns=rename_map, inplace=True)
    df_valid_seg.rename(columns=rename_map, inplace=True)
    df_test1_seg.rename(columns=rename_map, inplace=True)
    df_test2_seg.rename(columns=rename_map, inplace=True)

    # Clean specified text columns
    for col in ['Imaging and Studies', 'Pertinent Results']:
        for df in [df_train_seg, df_valid_seg, df_test1_seg, df_test2_seg]:
            df[col] = df[col].apply(clean_text)

    # Optionally, update headers in certain sections if present
    for col, new_text in [('patients_admissions', 'Patient admissions:'), ('transfer_summary', 'Transfer summary:')]:
        for df in [df_train_seg, df_valid_seg, df_test1_seg, df_test2_seg]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: re.sub(rf'^{col}:', new_text, x))

    # Fill any remaining missing values with empty strings
    for df in [df_train_seg, df_valid_seg, df_test1_seg, df_test2_seg]:
        df.fillna('', inplace=True)

    # Load target CSVs and replace corresponding columns
    target_train = load_csv(base_dir, "train", "discharge_target.csv.gz", **read_kwargs)
    target_valid = load_csv(base_dir, "valid", "discharge_target.csv.gz", **read_kwargs)
    target_test1 = load_csv(base_dir, "test_phase_1", "discharge_target.csv.gz", **read_kwargs)

    df_train_seg['brief_hospital_course'] = target_train['brief_hospital_course']
    df_train_seg['discharge_instructions'] = target_train['discharge_instructions']
    df_valid_seg['brief_hospital_course'] = target_valid['brief_hospital_course']
    df_valid_seg['discharge_instructions'] = target_valid['discharge_instructions']
    df_test1_seg['brief_hospital_course'] = target_test1['brief_hospital_course']
    df_test1_seg['discharge_instructions'] = target_test1['discharge_instructions']

    # Merge train and valid segmented DataFrames
    df_train_seg = pd.concat([df_train_seg, df_valid_seg], ignore_index=True)

    # Save the final DataFrames
    df_train_seg.to_csv(os.path.join(dataset_dir, "df_discharge_train_segmented.csv"), index=False)
    df_test1_seg.to_csv(os.path.join(dataset_dir, "df_discharge_test_phase_1_segmented.csv"), index=False)
    df_test2_seg.to_csv(os.path.join(dataset_dir, "df_discharge_test_phase_2_segmented.csv"), index=False)

    logger.info("Processing completed and files saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIMIC discharge data.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/datadisk/mimic/discharge",
        help="Base directory for MIMIC discharge data"
    )
    args = parser.parse_args()
    main(args)
