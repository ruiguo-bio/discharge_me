import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# load the relevant mimic-iv data
mimic_folder = '/mnt/datadisk/mimic/mimic-iv-2.2/hosp'
# Tables need to use:
# patients
# 'anchor_age','anchor_year','gender'
# admissions
# 'hadm_id','admittime','admission_type','race'
# transfers
# 'eventtype','hadm_id','intime'
# prescriptions
# 'drug','prod_strength', 'dose_val_rx','dose_unit_rx','starttime','hadm_id'
# diagnoses_icd
# 'hadm_id','icd_code','icd_version'
# d_icd_diagnoses
# 'icd_code','long_title'


patients = pd.read_csv(f'{mimic_folder}/patients.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
admissions = pd.read_csv(f'{mimic_folder}/admissions.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
transfers = pd.read_csv(f'{mimic_folder}/transfers.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
transfers['hadm_id'] = transfers['hadm_id'].astype('int', errors='ignore')
# prescriptions = pd.read_csv(f'{mimic_folder}/prescriptions.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
diagnoses_icd = pd.read_csv(f'{mimic_folder}/diagnoses_icd.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
d_icd_diagnoses = pd.read_csv(f'{mimic_folder}/d_icd_diagnoses.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')


# merge the patients and admissions dataframes by 'subject_id'
patients_admissions = patients.merge(admissions, on='subject_id')
# anchor_age is the age of the patient in the year of 'anchor_year'
# 'admittime' is the time of admission time, in the string format 'YYYY-MM-DD HH:MM:SS'
# can calculate the age of the patient at the time of admission
patients_admissions['admittime'] = pd.to_datetime(patients_admissions['admittime'])
# the patient actual age at the time of admission can be calculated by the difference between the year of 'admittime' and 'anchor_year' and 'anchor_age'
patients_admissions['age'] = patients_admissions['admittime'].dt.year - patients_admissions['anchor_year'] + patients_admissions['anchor_age']

# get the diagnosis of the patient's one admission
# merge the diagnoses_icd and d_icd_diagnoses dataframes by 'icd_code' and 'icd_version', get the 'long_title' column
diagnoses = diagnoses_icd.merge(d_icd_diagnoses[['icd_code', 'long_title']], on='icd_code')
# merge the patients_admissions and diagnoses dataframes by 'hadm_id'
patients_admissions_diagnoses = patients_admissions.merge(diagnoses, on='hadm_id')

# get the prescription of the patient's one admission
# merge the patients_admissions and prescriptions dataframes by 'hadm_id'
# patients_admissions_prescriptions = patients_admissions.merge(prescriptions, on='hadm_id')

# get the transfer of the patient's one admission
# merge the patients_admissions and transfers dataframes by 'hadm_id'
patients_admissions_transfers = patients_admissions.merge(transfers, on='hadm_id')

# for each unique 'hadm_id'
# create a summary of the patients admission
# the summary should include the patient age, sex, race, admittime, admission_location

# this is applied to each row of the patients_admissions_transfers dataframe
def admission_summary(row):
    this_dict =  {
        'Age': row['age'],
        'Sex': row['gender'],
        'race': row['race'],
        # timestamp to string
        'admittime': str(row['admittime']),
        # NaN values in the 'deathtime' column are omitted, only keep the non-NaN values
        'deathtime': str(row['deathtime']) if not pd.isnull(row['deathtime']) else '',
        'admission_type': row['admission_type'],
        'admission_location': row['admission_location']

    }
    # remove empty deathtime
    if this_dict['deathtime'] == '':
        del this_dict['deathtime']


    # concatenate the key and values in the dictionary into a single string
    return ', '.join([f"{k}: {v}" for k, v in this_dict.items()])


# apply the admission_summary function to each row of the patients_admissions_transfers dataframe
patients_admissions = patients_admissions.set_index('hadm_id')

# Apply the function row-wise using .apply(axis=1) and convert to dict
print('Applying admission_summary function to each row of the patients_admissions dataframe...')
patients_admissions_dict = patients_admissions.progress_apply(admission_summary, axis=1).to_dict()


# for each unique 'hadm_id', create a summary of the patients transfers

# this is applied to each row of the patients_admissions_transfers dataframe
# sort the transfers by 'intime' for each 'hadm_id'
# then create a summary column with the 'intime' and 'eventtype' columns
def transfer_summary(group):
    sorted_group = group.sort_values(by='intime')

    # Initialize an empty list to store each "eventtype: intime" string
    event_strings = []

    # Iterate through each row in the sorted group
    for index, row in enumerate(sorted_group.itertuples(), 1):
        # Format the "eventtype: intime" string for the current row

        event_string = f"{str(index)}: {row.intime} {row.eventtype}"
        # Append the formatted string to the list
        event_strings.append(event_string)

    # Join all the event strings into a single string, separated by ", "
    result_string = ", ".join(event_strings)
    return result_string

# find the total duration of the patient
def transfer_total_duration(group):
    sorted_group = group.sort_values(by='intime')
    # the first row 'intime' is the first time the patient is admitted
    # the last row 'intime' is the last time the patient is discharged
    # the difference between the last and first row 'intime' is the total duration of the patient
    # first convert str to datetime
    admit_time = pd.to_datetime(sorted_group.iloc[0]['intime'])
    discharge_time = pd.to_datetime(sorted_group.iloc[-1]['intime'])
    total_duration = discharge_time - admit_time
    # convert the total duration to hours
    total_duration = total_duration.total_seconds() / 3600
    return total_duration
# apply the transfer_total_duration function to each row of the patients_admissions_transfers dataframe
# group the results by 'hadm_id'
print('Applying transfer_total_duration function to each row of the patients_admissions_transfers dataframe...')
transfer_total_duration_dict = patients_admissions_transfers.groupby('hadm_id').progress_apply(transfer_total_duration).to_dict()

# apply the transfer_summary function to each row of the patients_admissions_transfers dataframe
# group the results by 'hadm_id'
print('Applying transfer_summary function to each row of the patients_admissions_transfers dataframe...')
transfer_summary_dict = patients_admissions_transfers.groupby('hadm_id').progress_apply(transfer_summary).to_dict()


# for each unique 'hadm_id', create a summary of the patients prescriptions

# this is applied to each row of the patients_admissions_prescriptions dataframe
# sort the prescriptions by 'starttime' for each 'hadm_id'
# then create a summary column with the 'starttime', 'drug','prod_strength', 'dose_val_rx','dose_unit_rx'columns
def prescription_summary(group):
    sorted_group = group.sort_values(by='starttime')

    # Initialize an empty list to store each "eventtype: intime" string
    event_strings = []

    # Iterate through each row in the sorted group
    for index, row in enumerate(sorted_group.itertuples(), 1):
        # record keys 'starttime', 'drug', 'prod_strength', 'dose_val_rx', 'dose_unit_rx'
        # start with the 1. 2. 3. 4. 5. for each row from the index
        event_string = f"{index}: start time: {row.starttime}, {row.drug} {row.prod_strength}, {row.dose_val_rx} {row.dose_unit_rx}"
        # Append the formatted string to the list
        event_strings.append(event_string)

    # Join all the event strings into a single string, separated by ", "
    result_string = ", ".join(event_strings)
    return result_string
# apply the prescription_summary function to each row of the patients_admissions_prescriptions dataframe
# group the results by 'hadm_id'
# print('Applying prescription_summary function to each row of the patients_admissions_prescriptions dataframe...')
# prescription_dict = patients_admissions_prescriptions.groupby('hadm_id').progress_apply(prescription_summary).to_dict()

# for each unique 'hadm_id', create a summary of the patients diagnoses

# this is applied to each row of the patients_admissions_diagnoses dataframe
# then create a summary column with the 'long_title' column
def diagnosis_summary(group):
    # Assuming 'long_title' is the column with the diagnosis descriptions
    # And you want to concatenate these descriptions for each 'hadm_id'

    # Initialize an empty list to store each "diagnosis" string
    diagnosis_strings = []

    # Iterate through each row in the group, at most 10 rows
    for index, row in enumerate(group.itertuples(), 1):
        # Format the string with the current index and the 'long_title' of the diagnosis
        diagnosis_string = f"{index}: {row.long_title}"
        # Append the formatted string to the list
        diagnosis_strings.append(diagnosis_string)

        if index == 10:
            break
    # Join all the diagnosis strings into a single string, separated by ", "
    result_string = ", ".join(diagnosis_strings)
    return result_string


# apply the diagnosis_summary function to each row of the patients_admissions_diagnoses dataframe
# group the results by 'hadm_id'
print('Applying diagnosis_summary function to each row of the patients_admissions_diagnoses dataframe...')
diagnoses_dict = diagnoses.groupby('hadm_id').progress_apply(diagnosis_summary).to_dict()

# save the dict to files
import json
print('Saving the dict to files...')
output_folder = '/mnt/datadisk/mimic/discharge/dataset'
# save 'patients_admissions_dict.json'

with open(f'{output_folder}/patients_admissions_dict.json', 'w') as f:
    json.dump(patients_admissions_dict, f)
with open(f'{output_folder}/transfer_summary_dict.json', 'w') as f:
    json.dump(transfer_summary_dict, f)
# with open('prescription_dict.json', 'w') as f:
#     json.dump(prescription_dict, f)
with open(f'{output_folder}/diagnoses_dict.json', 'w') as f:
    json.dump(diagnoses_dict, f)
with open(f'{output_folder}/transfer_total_duration_dict.json', 'w') as f:
    json.dump(transfer_total_duration_dict, f)
