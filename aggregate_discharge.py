import argparse
import pandas as pd
from tqdm import tqdm
import json

tqdm.pandas()

def process_mimic_data(mimic_folder: str, output_folder: str):
    """
    Processes MIMIC-IV data to generate summaries for admissions, transfers, prescriptions, and diagnoses.

    Args:
        mimic_folder (str): Path to the folder containing MIMIC-IV data.
        output_folder (str): Path to the folder where output JSON files will be saved.
    """
    # Load relevant MIMIC-IV tables
    print("Loading MIMIC-IV data...")
    patients = pd.read_csv(f'{mimic_folder}/patients.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    admissions = pd.read_csv(f'{mimic_folder}/admissions.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    transfers = pd.read_csv(f'{mimic_folder}/transfers.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    diagnoses_icd = pd.read_csv(f'{mimic_folder}/diagnoses_icd.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
    d_icd_diagnoses = pd.read_csv(f'{mimic_folder}/d_icd_diagnoses.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')

    # Ensure 'hadm_id' in transfers is treated as integer where possible
    transfers['hadm_id'] = transfers['hadm_id'].astype('int', errors='ignore')

    # Merge patients and admissions data
    patients_admissions = patients.merge(admissions, on='subject_id')
    patients_admissions['admittime'] = pd.to_datetime(patients_admissions['admittime'])
    patients_admissions['age'] = patients_admissions['admittime'].dt.year - patients_admissions['anchor_year'] + patients_admissions['anchor_age']

    # Add diagnosis information
    diagnoses = diagnoses_icd.merge(d_icd_diagnoses[['icd_code', 'long_title']], on='icd_code')
    patients_admissions_diagnoses = patients_admissions.merge(diagnoses, on='hadm_id')

    # Add transfer information
    patients_admissions_transfers = patients_admissions.merge(transfers, on='hadm_id')

    # Function to generate admission summary
    def admission_summary(row):
        summary = {
            'Age': row['age'],
            'Sex': row['gender'],
            'Race': row['race'],
            'Admittime': str(row['admittime']),
            'Deathtime': str(row['deathtime']) if not pd.isnull(row['deathtime']) else None,
            'Admission Type': row['admission_type'],
            'Admission Location': row['admission_location']
        }
        # Remove keys with None values
        summary = {k: v for k, v in summary.items() if v is not None}
        return ', '.join([f"{k}: {v}" for k, v in summary.items()])

    # Generate admission summaries
    print("Generating admission summaries...")
    patients_admissions_dict = patients_admissions.set_index('hadm_id').progress_apply(admission_summary, axis=1).to_dict()

    # Function to generate transfer summary
    def transfer_summary(group):
        sorted_group = group.sort_values(by='intime')
        event_strings = [
            f"{i}: {row.intime} {row.eventtype}"
            for i, row in enumerate(sorted_group.itertuples(), 1)
        ]
        return ", ".join(event_strings)

    # Function to calculate total transfer duration
    def transfer_total_duration(group):
        sorted_group = group.sort_values(by='intime')
        admit_time = pd.to_datetime(sorted_group.iloc[0]['intime'])
        discharge_time = pd.to_datetime(sorted_group.iloc[-1]['intime'])
        total_duration = (discharge_time - admit_time).total_seconds() / 3600
        return total_duration

    # Generate transfer summaries and durations
    print("Generating transfer summaries and durations...")
    transfer_summary_dict = patients_admissions_transfers.groupby('hadm_id').progress_apply(transfer_summary).to_dict()
    transfer_total_duration_dict = patients_admissions_transfers.groupby('hadm_id').progress_apply(transfer_total_duration).to_dict()

    # Function to generate diagnosis summary
    def diagnosis_summary(group):
        diagnosis_strings = [
            f"{i}: {row.long_title}"
            for i, row in enumerate(group.itertuples(), 1)
            if i <= 10  # Limit to top 10 diagnoses
        ]
        return ", ".join(diagnosis_strings)

    # Generate diagnosis summaries
    print("Generating diagnosis summaries...")
    diagnoses_dict = patients_admissions_diagnoses.groupby('hadm_id').progress_apply(diagnosis_summary).to_dict()

    # Save results to JSON files
    print("Saving results to JSON files...")
    with open(f'{output_folder}/patients_admissions_dict.json', 'w') as f:
        json.dump(patients_admissions_dict, f)
    with open(f'{output_folder}/transfer_summary_dict.json', 'w') as f:
        json.dump(transfer_summary_dict, f)
    with open(f'{output_folder}/diagnoses_dict.json', 'w') as f:
        json.dump(diagnoses_dict, f)
    with open(f'{output_folder}/transfer_total_duration_dict.json', 'w') as f:
        json.dump(transfer_total_duration_dict, f)

    print("Processing complete.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process MIMIC-IV data to generate summaries.")
    parser.add_argument(
        "--mimic_folder",
        type=str,
        default='/mnt/datadisk/mimic/mimic-iv-2.2/hosp',
        help="Path to the folder containing MIMIC-IV data."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default='/mnt/datadisk/mimic/discharge/dataset',
        help="Path to the folder where output JSON files will be saved."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the processing function
    process_mimic_data(args.mimic_folder, args.output_folder)