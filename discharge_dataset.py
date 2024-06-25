import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import re
import json
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler('token_cost.log')
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
df_discharge_train = pd.read_csv('/mnt/datadisk/mimic/discharge/train/discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
df_discharge_valid = pd.read_csv('/mnt/datadisk/mimic/discharge/valid/discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
df_discharge_test_phase_1 = pd.read_csv('/mnt/datadisk/mimic/discharge/test_phase_1/discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
df_discharge_test_phase_2 = pd.read_csv('/mnt/datadisk/mimic/discharge/test_phase_2/discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
#
#
#
print('Loading the dict from files...')
data_folder = '/mnt/datadisk/mimic/discharge/dataset'
with open(f'{data_folder}/patients_admissions_dict.json', 'r') as f:
    patients_admissions_dict = json.load(f)
with open(f'{data_folder}/transfer_summary_dict.json', 'r') as f:
    transfer_summary_dict = json.load(f)
# with open('prescription_dict.json', 'r') as f:
#     prescription_dict = json.load(f)
with open(f'{data_folder}/diagnoses_dict.json', 'r') as f:
    diagnoses_dict = json.load(f)
#
#
# associate those four dict with the discharge dataframes by 'hadm_id'
# note that the 'hadm_id' in those dict is str type, while the 'hadm_id' in the discharge dataframes is int type
# need to compare the same type
df_discharge_train['hadm_id'] = df_discharge_train['hadm_id'].astype(str)
df_discharge_valid['hadm_id'] = df_discharge_valid['hadm_id'].astype(str)
df_discharge_test_phase_1['hadm_id'] = df_discharge_test_phase_1['hadm_id'].astype(str)
df_discharge_test_phase_2['hadm_id'] = df_discharge_test_phase_2['hadm_id'].astype(str)

df_patients_admissions = pd.DataFrame(patients_admissions_dict.items(), columns=['hadm_id', 'patients_admissions'])
df_transfer_summary = pd.DataFrame(transfer_summary_dict.items(), columns=['hadm_id', 'transfer_summary'])
# df_prescription = pd.DataFrame(prescription_dict.items(), columns=['hadm_id', 'prescription'])
df_diagnoses = pd.DataFrame(diagnoses_dict.items(), columns=['hadm_id', 'diagnoses'])


df_discharge_train = df_discharge_train.merge(df_patients_admissions, on='hadm_id',how='left')
# set NaN to empty string for 'patients_admissions' column
df_discharge_train['patients_admissions'] = df_discharge_train['patients_admissions'].fillna('')

df_discharge_valid = df_discharge_valid.merge(df_patients_admissions, on='hadm_id',how='left')
# set NaN to empty string for 'patients_admissions' column
df_discharge_valid['patients_admissions'] = df_discharge_valid['patients_admissions'].fillna('')

df_discharge_test_phase_1 = df_discharge_test_phase_1.merge(df_patients_admissions, on='hadm_id',how='left')
# set NaN to empty string for 'patients_admissions' column
df_discharge_test_phase_1['patients_admissions'] = df_discharge_test_phase_1['patients_admissions'].fillna('')

df_discharge_test_phase_2 = df_discharge_test_phase_2.merge(df_patients_admissions, on='hadm_id',how='left')
# set NaN to empty string for 'patients_admissions' column
df_discharge_test_phase_2['patients_admissions'] = df_discharge_test_phase_2['patients_admissions'].fillna('')

# do the same on the 'transfer_summary', and  'diagnoses' columns
df_discharge_train = df_discharge_train.merge(df_transfer_summary, on='hadm_id',how='left')
df_discharge_train['transfer_summary'] = df_discharge_train['transfer_summary'].fillna('')
df_discharge_valid = df_discharge_valid.merge(df_transfer_summary, on='hadm_id',how='left')
df_discharge_valid['transfer_summary'] = df_discharge_valid['transfer_summary'].fillna('')
df_discharge_test_phase_1 = df_discharge_test_phase_1.merge(df_transfer_summary, on='hadm_id',how='left')
df_discharge_test_phase_1['transfer_summary'] = df_discharge_test_phase_1['transfer_summary'].fillna('')
df_discharge_test_phase_2 = df_discharge_test_phase_2.merge(df_transfer_summary, on='hadm_id',how='left')
df_discharge_test_phase_2['transfer_summary'] = df_discharge_test_phase_2['transfer_summary'].fillna('')

df_discharge_train = df_discharge_train.merge(df_diagnoses, on='hadm_id',how='left')
df_discharge_train['diagnoses'] = df_discharge_train['diagnoses'].fillna('')
df_discharge_valid = df_discharge_valid.merge(df_diagnoses, on='hadm_id',how='left')
df_discharge_valid['diagnoses'] = df_discharge_valid['diagnoses'].fillna('')
df_discharge_test_phase_1 = df_discharge_test_phase_1.merge(df_diagnoses, on='hadm_id',how='left')
df_discharge_test_phase_1['diagnoses'] = df_discharge_test_phase_1['diagnoses'].fillna('')
df_discharge_test_phase_2 = df_discharge_test_phase_2.merge(df_diagnoses, on='hadm_id',how='left')
df_discharge_test_phase_2['diagnoses'] = df_discharge_test_phase_2['diagnoses'].fillna('')
# concatenate those newly added columns to a new column "additional_info" in the discharge dataframes
# add the name such as 'patients admissions', 'transfer summary', 'diagnoses' before the content for each column
df_discharge_train['additional_info'] = df_discharge_train[['patients_admissions', 'transfer_summary',  'diagnoses']].apply(
    lambda x: '\n'.join([f"{k}: {v}" for k, v in x.items()]),
    axis=1
)

df_discharge_valid['additional_info'] = df_discharge_valid[['patients_admissions', 'transfer_summary', 'diagnoses']].apply(
    lambda x: '\n'.join([f"{k}: {v}" for k, v in x.items()]),
    axis=1
)

df_discharge_test_phase_1['additional_info'] = df_discharge_test_phase_1[['patients_admissions', 'transfer_summary', 'diagnoses']].apply(
    lambda x: '\n'.join([f"{k}: {v}" for k, v in x.items()]),
    axis=1
)

df_discharge_test_phase_2['additional_info'] = df_discharge_test_phase_2[['patients_admissions', 'transfer_summary', 'diagnoses']].apply(
    lambda x: '\n'.join([f"{k}: {v}" for k, v in x.items()]),
    axis=1
)

# drop the columns 'patients_admissions', 'transfer_summary', 'diagnoses' after concatenating them to the 'additional_info' column
df_discharge_train.drop(columns=['patients_admissions', 'transfer_summary', 'diagnoses'], inplace=True)
df_discharge_valid.drop(columns=['patients_admissions', 'transfer_summary', 'diagnoses'], inplace=True)
df_discharge_test_phase_1.drop(columns=['patients_admissions', 'transfer_summary', 'diagnoses'], inplace=True)
df_discharge_test_phase_2.drop(columns=['patients_admissions', 'transfer_summary', 'diagnoses'], inplace=True)

# split the text column by discarding the text before "Allergies:"
df_discharge_train['text'] = df_discharge_train['text'].str.extract(r'(Allergies:.*)', flags=re.DOTALL)
df_discharge_valid['text'] = df_discharge_valid['text'].str.extract(r'(Allergies:.*)', flags=re.DOTALL)
df_discharge_test_phase_1['text'] = df_discharge_test_phase_1['text'].str.extract(r'(Allergies:.*)', flags=re.DOTALL)
df_discharge_test_phase_2['text'] = df_discharge_test_phase_2['text'].str.extract(r'(Allergies:.*)', flags=re.DOTALL)
# concate the additional_info column to the text column
df_discharge_train['text'] = df_discharge_train[['additional_info','text' ]].apply(lambda x: '\n'.join(x), axis=1)
df_discharge_valid['text'] = df_discharge_valid[['additional_info','text' ]].apply(lambda x: '\n'.join(x), axis=1)
df_discharge_test_phase_1['text'] = df_discharge_test_phase_1[['additional_info','text' ]].apply(lambda x: '\n'.join(x), axis=1)
df_discharge_test_phase_2['text'] = df_discharge_test_phase_2[['additional_info','text' ]].apply(lambda x: '\n'.join(x), axis=1)

def segment_input_text(input_text, patterns):
    # Initialize a dictionary to hold the segmented content
    segmented_content = {key: "" for key in patterns.keys()}
    section_order = list(patterns.keys())  # List maintaining the order of sections

    # Preprocess the patterns to include optional colons and case-insensitive matching
    processed_patterns = {}
    for section, pattern in patterns.items():
        if isinstance(pattern, list):
            processed_patterns[section] = [re.compile(rf'^[\s_\-]*{re.escape(p.strip(":"))}[:]*\s*', re.IGNORECASE) for
                                           p in pattern]
        else:
            processed_patterns[section] = [
                re.compile(rf'^[\s_\-]*{re.escape(pattern.strip(":"))}[:]*\s*', re.IGNORECASE)]

    # Split the input text into lines for easier processing
    lines = input_text.split('\n')

    # Track the current section being processed
    current_section = None
    last_matched_index = -1  # To keep track of the last matched section index

    for line in lines:
        # if line == 'IMAGING':
        #     print('IMAGING')
        matched = False
        # Identify the section based on the start of the line matching any processed pattern
        for i, (section, patterns) in enumerate(processed_patterns.items()):
            if matched:
                break
            for compiled_pattern in patterns:
                if compiled_pattern.match(line):
                    if section == 'brief hospital course' or section == 'discharge instructions':
                        current_section = section
                        last_matched_index = i
                        matched = True
                        break

                    else:
                        # Only update the current section if it's the same or a subsequent one in the order
                        if i > last_matched_index and i < last_matched_index + 4:
                            current_section = section
                            last_matched_index = i
                            matched = True
                            break

        # Append the line to the correct section, either a new one or a continuing one
        if current_section:
            segmented_content[current_section] += line + '\n'

    return segmented_content

#
# # This adjusted function now will add any lines that don't match a new section directly into the last identified section,
# # ensuring that information isn't lost or misplaced.
#
#
# # split the df_discharge_train['brief_hospital_course'] into different sections
# # Define the patterns based on the revised dictionary structure
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
    "REVIEW OF SYSTEMS:": ["REVIEW OF SYSTEMS:",'ros'],
    "Past Medical History": ["Past Medical History:",'Oncologic History','Other Past Medical History'],
    "Social History": "Social History:",
    "Family History": "Family History:",
    "Physical Exam": ['physical examination','admission vitals','discharge vitals',"PHYSICAL EXAM:",'ADMISSION PHYSICAL EXAM','physical exam on admission','physical exam on discharge','admission physical examination','discharge physical examination','discharge PHYSICAL EXAM','discharge EXAM','admission EXAM','vital signs','exam on discharge','exam on admission'],
    "Pertinent Results": ["Pertinent Results:","labs on admission","labs on discharge","admission labs","discharge labs",'OTHER LABS','CSF Studies', "PERTINENT LABS",'microbiology','CBC w/ diff','Rheumatologic testing','ekg','ecg'],
    "Imaging and Studies": ["IMAGING:",'imaging/studies',"IMPRESSION", "STUDIES:","CXR","mri","echo","cat scan",'cta'],
    'brief hospital course': 'brief hospital course:',
    "admission medications": ["admission medications:","medications on admission"],
    "Discharge Medications": ["Discharge Medications:","medications on discharge"],

    "Discharge Disposition": "Discharge Disposition:",
    "Discharge Diagnosis": ["Discharge Diagnosis:",'primary diagnosis','secondary diagnosis'],
    "Discharge Condition": "Discharge Condition:",
    "Discharge Instructions": "Discharge Instructions:",
    "Followup Instructions": "Followup Instructions:",
    "Provider": "Provider:",
    "code status": "code status:",


}
#
#
#
import re

def find_section_patterns(text: str) -> set:
    pattern = re.compile(r'^\s*([\w\s]{1,30}):\s*', re.MULTILINE)
    found_patterns = set()
    matches = pattern.findall(text)
    for match in matches:
        if len(match.split()) <= 5:
            found_patterns.add(match.strip())
    return found_patterns

# convert column hadm_id to int
df_discharge_train['hadm_id'] = df_discharge_train['hadm_id'].astype(int)
df_discharge_valid['hadm_id'] = df_discharge_valid['hadm_id'].astype(int)
df_discharge_test_phase_1['hadm_id'] = df_discharge_test_phase_1['hadm_id'].astype(int)
df_discharge_test_phase_2['hadm_id'] = df_discharge_test_phase_2['hadm_id'].astype(int)

input_text =  df_discharge_train[df_discharge_train['hadm_id'] == 21855857]['text'].values[0]
segment_input_text(input_text, patterns)
# sample 3000 rows to test
# df_discharge_train_sample = df_discharge_train.sample(3000)
# apply segment_input_text_with_previous_section_fallback on the df_discharge_train_sample['text']
df_discharge_train['segmented_content'] = df_discharge_train['text'].progress_apply(segment_input_text, patterns=patterns)
df_discharge_valid['segmented_content'] = df_discharge_valid['text'].progress_apply(segment_input_text, patterns=patterns)
df_discharge_test_phase_1['segmented_content'] = df_discharge_test_phase_1['text'].progress_apply(segment_input_text, patterns=patterns)
df_discharge_test_phase_2['segmented_content'] = df_discharge_test_phase_2['text'].progress_apply(segment_input_text, patterns=patterns)

df_discharge_train['segmented_content_length'] = df_discharge_train['segmented_content'].apply(
    lambda x: {k: len(v) for k, v in x.items()})
df_discharge_valid['segmented_content_length'] = df_discharge_valid['segmented_content'].apply(
    lambda x: {k: len(v) for k, v in x.items()})
df_discharge_test_phase_1['segmented_content_length'] = df_discharge_test_phase_1['segmented_content'].apply(
    lambda x: {k: len(v) for k, v in x.items()})

df_discharge_test_phase_2['segmented_content_length'] = df_discharge_test_phase_2['segmented_content'].apply(
    lambda x: {k: len(v) for k, v in x.items()})
# find the length distribution of each section in the df_discharge_train['segmented_content_length']
# first convert the 'segmented_content_length' column to a list of dicts
data_for_analysis = []
for index, row in df_discharge_train.iterrows():
    data_for_analysis.append(row['segmented_content_length'])

lengths_df = pd.DataFrame(data_for_analysis)
lengths_df.fillna(0, inplace=True)
# output the statistics of the length of each section plus the 90% quantile
# save to xlsx
lengths_df.describe([.25, .5, .75,.9,.95]).to_excel('lengths_df_statistics.xlsx')
# cap the length of each section in df_discharge_train['segmented_content'] to the 90% quantile
# first calculate the 95% quantile of each section
quantile_90 = lengths_df.quantile(0.90)
# then cap the length of each section in df_discharge_train['segmented_content'] to the 90% quantile
df_discharge_train['segmented_content'] = df_discharge_train['segmented_content'].apply(
    lambda x: {k: v[:int(quantile_90[k])] for k, v in x.items()})
# do the same for df_discharge_valid and df_discharge_test_phase_1, df_discharge_test_phase_2
df_discharge_valid['segmented_content'] = df_discharge_valid['segmented_content'].apply(
    lambda x: {k: v[:int(quantile_90[k])] for k, v in x.items()})
df_discharge_test_phase_1['segmented_content'] = df_discharge_test_phase_1['segmented_content'].apply(
    lambda x: {k: v[:int(quantile_90[k])] for k, v in x.items()})
df_discharge_test_phase_2['segmented_content'] = df_discharge_test_phase_2['segmented_content'].apply(
    lambda x: {k: v[:int(quantile_90[k])] for k, v in x.items()})

# for the column 'segmented_content', change the dict key / value to the column name / value for
# do it the same on train,valid,test_phase_1,test_phase_2
df_expanded = df_discharge_train['segmented_content'].apply(pd.Series)

# Join the new columns back to the original DataFrame
df_discharge_train_segmented = df_discharge_train.join(df_expanded)

# Optionally drop the original 'segmented_content' column if no longer needed
df_discharge_train_segmented.drop(columns=['segmented_content'], inplace=True)
# drop the 'text' column
df_discharge_train_segmented.drop(columns=['text'], inplace=True)

# do the same for df_discharge_valid_segmented, df_discharge_test_phase_1_segmented, df_discharge_test_phase_2_segmented
df_expanded = df_discharge_valid['segmented_content'].apply(pd.Series)
df_discharge_valid_segmented = df_discharge_valid.join(df_expanded)
df_discharge_valid_segmented.drop(columns=['segmented_content','text'], inplace=True)

df_expanded = df_discharge_test_phase_1['segmented_content'].apply(pd.Series)
df_discharge_test_phase_1_segmented = df_discharge_test_phase_1.join(df_expanded)
df_discharge_test_phase_1_segmented.drop(columns=['segmented_content','text'], inplace=True)

df_expanded = df_discharge_test_phase_2['segmented_content'].apply(pd.Series)
df_discharge_test_phase_2_segmented = df_discharge_test_phase_2.join(df_expanded)
df_discharge_test_phase_2_segmented.drop(columns=['segmented_content','text'], inplace=True)


# read radiology.csv.gz
df_radiology = pd.read_csv('/mnt/datadisk/mimic/discharge/train/radiology.csv.gz', compression='gzip')

# concat the 'text' column in df_radiology by the same hadm_id
df_radiology = df_radiology.groupby('hadm_id')['text'].apply(' '.join).reset_index()
# rename 'text' to 'radiology'
df_radiology.rename(columns={'text':'radiology'}, inplace=True)
# fill na with empty string
df_radiology.fillna('', inplace=True)
# merge the df_radiology with df_discharge_train_segmented
df_discharge_train_segmented = pd.merge(df_discharge_train_segmented, df_radiology, on='hadm_id', how='left')


df_radiology = pd.read_csv('/mnt/datadisk/mimic/discharge/valid/radiology.csv.gz', compression='gzip')
df_radiology = df_radiology.groupby('hadm_id')['text'].apply(' '.join).reset_index()
# rename 'text' to 'radiology'
df_radiology.rename(columns={'text':'radiology'}, inplace=True)
# fill na with empty string
df_radiology.fillna('', inplace=True)
# merge the df_radiology with df_discharge_valid_segmented
df_discharge_valid_segmented = pd.merge(df_discharge_valid_segmented, df_radiology, on='hadm_id', how='left')
# merge the df_radiology with df_discharge_test_phase_1_segmented

df_radiology = pd.read_csv('/mnt/datadisk/mimic/discharge/test_phase_1/radiology.csv.gz', compression='gzip')
df_radiology = df_radiology.groupby('hadm_id')['text'].apply(' '.join).reset_index()
# rename 'text' to 'radiology'
df_radiology.rename(columns={'text':'radiology'}, inplace=True)
# fill na with empty string
df_radiology.fillna('', inplace=True)
df_discharge_test_phase_1_segmented = pd.merge(df_discharge_test_phase_1_segmented, df_radiology, on='hadm_id', how='left')
# merge the df_radiology with df_discharge_test_phase_2_segmented
df_radiology = pd.read_csv('/mnt/datadisk/mimic/discharge/test_phase_2/radiology.csv.gz', compression='gzip')
df_radiology = df_radiology.groupby('hadm_id')['text'].apply(' '.join).reset_index()
# rename 'text' to 'radiology'
df_radiology.rename(columns={'text':'radiology'}, inplace=True)
# fill na with empty string
df_radiology.fillna('', inplace=True)
df_discharge_test_phase_2_segmented = pd.merge(df_discharge_test_phase_2_segmented, df_radiology, on='hadm_id', how='left')


# find the max word count of df_discharge_train_segmented['Imaging and Studies']
# first calculate the word count
df_discharge_train_segmented['Imaging and Studies_word_count'] = df_discharge_train_segmented['Imaging and Studies'].apply(lambda x: len(x.split()))
# find the max word count
max_word_count = df_discharge_train_segmented['Imaging and Studies_word_count'].max()


# find the rows df_discharge_train_segmented['radiology'] is not string, show the first 5 rows
# df_discharge_train_segmented[~df_discharge_train_segmented['radiology'].apply(lambda x: isinstance(x, str))].head()

# cap the df_discharge_train_segmented['radiology'] to at most max_word_count
# if empty, just keep as it is
df_discharge_train_segmented['radiology'] = df_discharge_train_segmented['radiology'].apply(lambda x: x if x == '' else ' '.join(x.split()[:max_word_count]))
df_discharge_valid_segmented['radiology'] = df_discharge_valid_segmented['radiology'].apply(lambda x: x if x == '' else ' '.join(x.split()[:max_word_count]))
df_discharge_test_phase_1_segmented['radiology'] = df_discharge_test_phase_1_segmented['radiology'].apply(lambda x: x if x == '' else ' '.join(x.split()[:max_word_count]))
df_discharge_test_phase_2_segmented['radiology'] = df_discharge_test_phase_2_segmented['radiology'].apply(lambda x: x if x == '' else ' '.join(x.split()[:max_word_count]))

# if Imaging and Studies is empty, set it to radiology

df_discharge_train_segmented['Imaging and Studies'] = df_discharge_train_segmented.apply(lambda x: x['radiology'] if x['Imaging and Studies'] == '' else x['Imaging and Studies'], axis=1)
df_discharge_valid_segmented['Imaging and Studies'] = df_discharge_valid_segmented.apply(lambda x: x['radiology'] if x['Imaging and Studies'] == '' else x['Imaging and Studies'], axis=1)
df_discharge_test_phase_1_segmented['Imaging and Studies'] = df_discharge_test_phase_1_segmented.apply(lambda x: x['radiology'] if x['Imaging and Studies'] == '' else x['Imaging and Studies'], axis=1)
df_discharge_test_phase_2_segmented['Imaging and Studies'] = df_discharge_test_phase_2_segmented.apply(lambda x: x['radiology'] if x['Imaging and Studies'] == '' else x['Imaging and Studies'], axis=1)


# drop 'radiology'
df_discharge_train_segmented.drop(columns=['radiology'], inplace=True)
df_discharge_valid_segmented.drop(columns=['radiology'], inplace=True)
df_discharge_test_phase_1_segmented.drop(columns=['radiology'], inplace=True)
df_discharge_test_phase_2_segmented.drop(columns=['radiology'], inplace=True)







# drop the 'brief hospital course' and 'Discharge Instructions' column
# df_discharge_train_segmented.drop(columns=['brief hospital course', 'Discharge Instructions'], inplace=True)
# df_discharge_valid_segmented.drop(columns=['brief hospital course', 'Discharge Instructions'], inplace=True)
# df_discharge_test_phase_1_segmented.drop(columns=['brief hospital course', 'Discharge Instructions'], inplace=True)
# df_discharge_test_phase_2_segmented.drop(columns=['brief hospital course', 'Discharge Instructions'], inplace=True)
# rename 'brief hospital course' to 'brief_hospital_course' and 'Discharge Instructions' to 'discharge_instructions'
df_discharge_train_segmented.rename(columns={'brief hospital course': 'brief_hospital_course', 'Discharge Instructions': 'discharge_instructions'}, inplace=True)
df_discharge_valid_segmented.rename(columns={'brief hospital course': 'brief_hospital_course', 'Discharge Instructions': 'discharge_instructions'}, inplace=True)
df_discharge_test_phase_1.rename(columns={'brief hospital course': 'brief_hospital_course', 'Discharge Instructions': 'discharge_instructions'}, inplace=True)
df_discharge_test_phase_2_segmented.rename(columns={'brief hospital course': 'brief_hospital_course', 'Discharge Instructions': 'discharge_instructions'}, inplace=True)


pattern = r"^___ \d{2}:\d{2}(AM|PM) [A-Za-z]+\s*$"

# Function to clean a single text entry
def clean_text(text):
    # Remove lines that strictly match the pattern
    cleaned_text = re.sub(pattern, "", text, flags=re.MULTILINE)
    # Clean up potential empty lines left after removal
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
    return cleaned_text.strip()

df_discharge_train_segmented['Imaging and Studies'] = df_discharge_train_segmented['Imaging and Studies'].apply(clean_text)
df_discharge_valid_segmented['Imaging and Studies'] = df_discharge_valid_segmented['Imaging and Studies'].apply(clean_text)
df_discharge_test_phase_1_segmented['Imaging and Studies'] = df_discharge_test_phase_1_segmented['Imaging and Studies'].apply(clean_text)
df_discharge_test_phase_2_segmented['Imaging and Studies'] = df_discharge_test_phase_2_segmented['Imaging and Studies'].apply(clean_text)

# apply to column "Pertinent Results"
df_discharge_train_segmented['Pertinent Results'] = df_discharge_train_segmented['Pertinent Results'].apply(clean_text)
df_discharge_valid_segmented['Pertinent Results'] = df_discharge_valid_segmented['Pertinent Results'].apply(clean_text)
df_discharge_test_phase_1_segmented['Pertinent Results'] = df_discharge_test_phase_1_segmented['Pertinent Results'].apply(clean_text)
df_discharge_test_phase_2_segmented['Pertinent Results'] = df_discharge_test_phase_2_segmented['Pertinent Results'].apply(clean_text)

# replace the first word of column 'patients_admissions' from "patients_admissions" to 'Patient admissions'
df_discharge_train_segmented['patients_admissions'] = df_discharge_train_segmented['patients_admissions'].apply(lambda x: re.sub(r'^patients_admissions:', 'Patient admissions:', x))
df_discharge_valid_segmented['patients_admissions'] = df_discharge_valid_segmented['patients_admissions'].apply(lambda x: re.sub(r'^patients_admissions:', 'Patient admissions:', x))
df_discharge_test_phase_1_segmented['patients_admissions'] = df_discharge_test_phase_1_segmented['patients_admissions'].apply(lambda x: re.sub(r'^patients_admissions:', 'Patient admissions:', x))
df_discharge_test_phase_2_segmented['patients_admissions'] = df_discharge_test_phase_2_segmented['patients_admissions'].apply(lambda x: re.sub(r'^patients_admissions:', 'Patient admissions:', x))

# replace the first word of column 'transfer_summary' from "transfer_summary" to 'Transfer summary'
df_discharge_train_segmented['transfer_summary'] = df_discharge_train_segmented['transfer_summary'].apply(lambda x: re.sub(r'^transfer_summary:', 'Transfer summary:', x))
df_discharge_valid_segmented['transfer_summary'] = df_discharge_valid_segmented['transfer_summary'].apply(lambda x: re.sub(r'^transfer_summary:', 'Transfer summary:', x))
df_discharge_test_phase_1_segmented['transfer_summary'] = df_discharge_test_phase_1_segmented['transfer_summary'].apply(lambda x: re.sub(r'^transfer_summary:', 'Transfer summary:', x))
df_discharge_test_phase_2_segmented['transfer_summary'] = df_discharge_test_phase_2_segmented['transfer_summary'].apply(lambda x: re.sub(r'^transfer_summary:', 'Transfer summary:', x))

# fill na to empty string
df_discharge_train_segmented.fillna('', inplace=True)
df_discharge_valid_segmented.fillna('', inplace=True)
df_discharge_test_phase_1_segmented.fillna('', inplace=True)
df_discharge_test_phase_2_segmented.fillna('', inplace=True)



df_discharge_train_target = pd.read_csv('/mnt/datadisk/mimic/discharge/train/discharge_target.csv.gz', compression='gzip')
df_discharge_valid_target = pd.read_csv('/mnt/datadisk/mimic/discharge/valid/discharge_target.csv.gz', compression='gzip')
df_discharge_test_phase_1_target = pd.read_csv('/mnt/datadisk/mimic/discharge/test_phase_1/discharge_target.csv.gz', compression='gzip')
df_discharge_test_phase_2_target = pd.read_csv('/mnt/datadisk/mimic/discharge/test_phase_2/discharge_target.csv.gz', compression='gzip')

# replace the 'brief_hospital_course' and 'discharge_instructions' with the target
df_discharge_train_segmented['brief_hospital_course'] = df_discharge_train_target['brief_hospital_course']
df_discharge_train_segmented['discharge_instructions'] = df_discharge_train_target['discharge_instructions']
df_discharge_valid_segmented['brief_hospital_course'] = df_discharge_valid_target['brief_hospital_course']
df_discharge_valid_segmented['discharge_instructions'] = df_discharge_valid_target['discharge_instructions']
df_discharge_test_phase_1_segmented['brief_hospital_course'] = df_discharge_test_phase_1_target['brief_hospital_course']
df_discharge_test_phase_1_segmented['discharge_instructions'] = df_discharge_test_phase_1_target['discharge_instructions']

# merge the df_discharge_train_segmented with df_discharge_valid_segmented, df_discharge_test_phase_1_segmented, df_discharge_test_phase_2_segmented
df_discharge_train_segmented = pd.concat([df_discharge_train_segmented, df_discharge_valid_segmented], ignore_index=True)


df_discharge_train_segmented.to_csv('/mnt/datadisk/mimic/discharge/dataset/df_discharge_train_segmented.csv', index=False)
df_discharge_test_phase_1_segmented.to_csv('/mnt/datadisk/mimic/discharge/dataset/df_discharge_test_phase_1_segmented', index=False)
df_discharge_test_phase_2_segmented.to_csv('/mnt/datadisk/mimic/discharge/dataset/df_discharge_test_phase_2_segmented', index=False)



