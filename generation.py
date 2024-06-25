import os
import json
import pandas as pd
import sys
import time

import re

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="llama3:8b-instruct-q8_0",timeout=180,verbose=True,num_predict=4000,num_ctx=8000,cache=False,)

df_discharge_train_segmented = pd.read_csv('/mnt/datadisk/mimic/discharge/dataset/df_discharge_train_segmented.csv')
df_discharge_test_phase_2_segmented = pd.read_csv('/mnt/datadisk/mimic/discharge/dataset/df_discharge_test_phase_2_segmented.csv')
# fill na with ''
df_discharge_train_segmented.fillna('', inplace=True)
df_discharge_test_phase_2_segmented.fillna('', inplace=True)
# concat columns 'Chief Complaint', 'diagnoses', 'History of Present Illness' as 'input_bhc_rag'

output_parser = StrOutputParser()
# fill in the structure
bhc_structure = '''{
  "Brief Hospital Course": {
    "Introduction": "Brief introduction including patient demographics, significant past medical history, and reason for hospitalization.",
    "Active Issues": "Details of the primary medical concerns addressed during the stay, including initial assessments and management actions.",
    "Chronic Issues (Optional)": "Management of known chronic conditions during the hospital stay.",
    "Transitional Issues (Optional)": "Specific follow-up actions recommended for post-discharge care.",
    "Additional Notes (Optional)": "Other pertinent information or considerations affecting patient care."
  }
}'''


bhc_template = ("""
You are tasked with drafting a 'Brief Hospital Course' section for a discharge letter as a medical professional. 
Utilize the structure from an example of a brief hospital course to guide your composition. 
The goal is to write a new, coherent brief hospital course for another patient based on the provided structured template.
The total word count for the brief hospital course should be {words} words. 

Instructions:
- Follow the JSON template provided to structure the new brief hospital course. 
Each section should be filled according to the relevant patient information.
- Omit the optional sections if they are not relevant to the patient's case.
- Omit the optional sections if the total word count is less than 100 words.
- Do not add a new section after Additional Notes. 
- Use placeholders '___' for any date, patient name and location.
- Use appropriate medical terminology and concise language to ensure clarity and professionalism.
- Do not be wordy, be concise if possible.
- Do not include the word "optional" in the result if they are included. If they are not included, just omit those sections.
- Do not copy patient information verbatim; paraphrase and use the structure template to fit in the details.
- All the section header must be from the template, not from the patient information.
- Do not fabricate details not present in the patient information. 
- Use section headers for each major medical issue, starting with a hashtag `#`, do not use `*` for section header.
- Use bullet points to highlight key actions, medication changes, or critical clinical decisions, starting with a hyphen `-`. Do not use `*` or `+`.
- Ensure that each major issue or condition has its own section header if there is enough content related to it, even if briefly mentioned.
- Write in a narrative style for each section, providing a detailed account of the patient's condition, treatment, and outcomes.
- Employ medical abbreviations and terminology appropriately to convey information efficiently.
- Start the output with 'Brief hospital course:\n'


Example structure for the brief hospital course: {structure}.
Patient information: {context}

""")


di_structure = '''{
  "Discharge Instructions": {
    "Greeting": "Dear [Title] ___,",
    "HospitalExperience": "It was a pleasure taking care of you at ___.",
    "AdmissionReason": {
       "Title: WHY WAS I ADMITTED TO THE HOSPITAL?",
       "Details": "[ReasonForAdmission]"
    },
    "InHospitalActivities": {
      "Title": "WHAT HAPPENED WHILE I WAS IN THE HOSPITAL?",
      "Details": "[ActivitiesDuringStay]"
    },
    "DischargeAdvice": {
      "Title": "WHAT SHOULD I DO WHEN I GO HOME?",
      "Instructions": "[PostDischargeInstructions]"
    },
    "Closing": "We wish you the best!",
    "CareTeam": "Your ____ Team"
  }
}

'''



di_template = ("""
You are tasked with drafting a 'Discharge Instructions' section for a patient's discharge letter as a medical professional. 
The instructions should succinctly summarize the key points of the patient's hospital stay and post-discharge care in a way that is clear and easy for the patient to follow.

Instructions:
- Use the JSON template provided to structure the discharge instructions
- Do not include explicit section headers in the final text, such as 'Greeting' or 'Hospital Experience'.
- Do not include any placeholder such as "[]" in the result.
- Include the title in the template.
- Integrate medication information narratively, mentioning specific medications only when discussing their relevance to the patient’s ongoing care and follow-up instructions.
- Do not list medications; instead, describe how they contribute to the patient’s treatment plan.
- The total word count should be around {words} words, focusing on essential instructions relevant to the patient's care.
- Use '___' to anonymize any date, patient name and location. 
- Clearly specify any medication changes, follow-up appointments, and additional care instructions using placeholders where specific details are to be inserted.
- Employ a professional yet empathetic tone to ensure clarity and approachability.
- Integrate medical terminology appropriately, ensuring it is understandable to a layperson.
- Start the output with a polite greeting and conclude with well-wishes or a thank you message.

Example structure for the discharge instructions: {structure}.
Patient information: {context}

""")






from tqdm import tqdm



def save_data(data, filename="saved_responses.json"):

    """Save the data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to file: {filename}")

# load the previous saved data
def load_data(filename="saved_responses.json"):
    """Load the data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Data loaded from file: {filename}")
    return data

def remove_special_sentences(text):
    text = re.sub(r'\[.*?\]', '___', text)
    text = re.sub(r'\([^a-zA-Z]*\)', '', text)
    text = re.sub(r'=', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'none', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(optional\)', '', text, flags=re.IGNORECASE)
    # Updated to ensure capturing all text before the last sentence
    pattern_last = r'(.*?)(\s*Please note that [^\.]*\.)$'
    match_last = re.match(pattern_last, text, re.DOTALL)  # Using re.DOTALL to include newline characters
    if match_last:
        text = match_last.group(1)  # This now correctly captures all text before the last sentence

    pattern_last = r'(.*?)(\s*Note: [^\.]*\.)$'
    match_last = re.match(pattern_last, text, re.DOTALL)  # Using re.DOTALL to include newline characters
    if match_last:
        text = match_last.group(1)  # This now correctly captures all text before the last sentence

    pattern_last = r'(.*?)(\s*I hope this [^\.]*\.)$'
    match_last = re.match(pattern_last, text, re.DOTALL)  # Using re.DOTALL to include newline characters
    if match_last:
        text = match_last.group(1)  # This now correctly captures all text before the last sentence

    pattern_last = r'(.*?)(\s*Please let me [^\.]*\.)$'
    match_last = re.match(pattern_last, text, re.DOTALL)  # Using re.DOTALL to include newline characters
    if match_last:
        text = match_last.group(1)  # This now correctly captures all text before the last sentence

    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s|(\n+)'
    # Use re.split with capturing parentheses to keep the delimiters (newlines)
    parts = re.split(sentence_endings, text.strip())

    # Initialize a list to collect sentences and their trailing characters (like newlines)
    sentences = []
    current_sentence = ''
    for part in parts:
        if part is None:
            continue
        # Check if the current part is a newline, if so, append it directly to the sentence
        if part.strip() == '':
            current_sentence += part
        else:
            # Add the non-newline text to the current sentence
            current_sentence += part
            # Append the complete sentence to the list and reset for the next sentence
            sentences.append(current_sentence)
            current_sentence = ''

    filtered_sentences = []
    # Define the patterns to search for
    # pattern_additional_notes = re.compile(r'\badditional note[s]?\b', re.IGNORECASE)
    # pattern_none = re.compile(r'\bnone\b', re.IGNORECASE)
    pattern_word_count = re.compile(r'\bword count\b', re.IGNORECASE)
    pattern_hospital_course = re.compile(r'\bhere is a\b|\bbased on the new\b|\bhere is the\b', re.IGNORECASE)
    pattern_strict_hospital_course = re.compile(r'\s*brief hospital course\s*', re.IGNORECASE)
    pattern_hash_introduction = re.compile(r'\s*#?\s*introduction:?\b', re.IGNORECASE)
    pattern_strict_discharge_instructions = re.compile(r'\s*discharge instructions\s*', re.IGNORECASE)

    for sentence in sentences:
        # Check conditions for deletion
        if (pattern_word_count.search(sentence) or
                pattern_strict_hospital_course.search(sentence) or
                pattern_hash_introduction.search(sentence) or pattern_strict_discharge_instructions.search(sentence)):
            continue  # Skip sentences that meet any deletion criteria

        # Add sentence to the list if it doesn't meet deletion criteria
        filtered_sentences.append(sentence)

    result = ' '.join(filtered_sentences)
    result = result.lstrip('\n')
    # Join the remaining sentences back into a single text
    return result

@chain
def generation_chain(doc,bhc=True):
    query = doc.page_content
    docs = vectorstore.similarity_search_with_score(query)
    docs = [d[0] for d in docs]
    if bhc:
        template = bhc_template
        fill_in_context = doc.metadata['patients_admissions'] + '\n\n' + doc.metadata[
            'Chief Complaint'] + '\n\n' + doc.metadata['History of Present Illness'] + '\n\n' + \
                          doc.metadata['Imaging and Studies'] + '\n\n' + \
                          doc.metadata['Past Medical History']
        structure = bhc_structure
    else:
        template = di_template
        # change brief_hospital_course to generated_brief_hospital_course after the generation of the brief hospital course
        fill_in_context = doc.metadata['patients_admissions'] + '\n\n' + doc.metadata[
            'brief_hospital_course'] + '\n\n' + doc.metadata['Discharge Medications'] + '\n\n' + \
                          doc.metadata['Discharge Disposition'] + '\n\n' + doc.metadata['Discharge Diagnosis'] + '\n\n' + \
                          doc.metadata['Discharge Condition'] + '\n\n' + doc.metadata['Followup Instructions']

        structure = di_structure

    prompt_fill_in = PromptTemplate(template=template, input_variables=["structure", "context", "words"])
    chain_fill_in = (prompt_fill_in | llm | output_parser)

    retrieved_top = docs[0].metadata['brief_hospital_course']
    retrieved_word_count = len(retrieved_top.split())

    # do not generate too long text
    if retrieved_word_count > 800:
        retrieved_word_count = 800


    # Call the single item processing method
    # record the time for this process
    prompt = prompt_fill_in.invoke(
        {"structure": structure, "context": fill_in_context, "words": retrieved_word_count}).text
    response = chain_fill_in.invoke(
        {"structure": structure, "context": fill_in_context, "words": retrieved_word_count})
    return response



def process_items(start_index, end_index,bhc):
    all_responses = {}

    # Adjust end_index to be within bounds
    start_index = max(0, start_index)
    if end_index < 0:
        end_index = len(df_discharge_test_phase_2_segmented)
    else:
        end_index = min(end_index, len(df_discharge_test_phase_2_segmented))
    result_folder = '/mnt/datadisk/mimic/discharge/result'
    if bhc:
        save_file_name = f"{result_folder}/bhc_{start_index}_{end_index}.json"
    else:
        save_file_name = f"{result_folder}/di_{start_index}_{end_index}.json"

    if os.path.exists(save_file_name):
        all_responses = load_data(save_file_name)

    # Process in steps of 'batch_size'
    for idx in tqdm(range(start_index, end_index), desc='Processing'):
        try:

            print(f"Processing index {idx}")

            input_data = df_document_test[idx]

            if bhc:
                target = input_data.metadata['brief_hospital_course']
            else:
                target = input_data.metadata['discharge_instructions']
            hadm_id = input_data.metadata['hadm_id']

            # Call the single item processing method
            # record the time for this process
            start_time = time.time()
            # prompt = prompt_fill_in.invoke({"structure":detailed_structure,"context":fill_in_context,"words":target_word_count}).text
            # response = chain_fill_in.invoke({"structure":detailed_structure,"context":fill_in_context,"words":target_word_count})
            response = generation_chain.invoke(input_data,bhc=bhc)
            cleaned_response = remove_special_sentences(response)
            end_time = time.time()
            #'hadm_id', 'generated_brief_hospital_course', 'target',
            #'response_word_count', 'target_word_counts'
            all_responses[str(idx)] = (int(hadm_id),cleaned_response,target)

            # Save the data to a file
            save_data(all_responses, save_file_name)
            print(f"Time taken to process index {idx}: {end_time - start_time} seconds")

        except Exception as e:
            print(f"An error occurred while processing index {idx}: {str(e)}")
            all_responses[idx] = None
            # Optionally, log error details or take corrective actions


all_responses = {}
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generation.py <start_index> <end_index> <bhc>")
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    # bhc or di generation
    bhc = int(sys.argv[3])

    # print
    print(f"Start index: {start_idx}")
    print(f"End index: {end_idx}")
    if bhc:
        print("Brief Hospital Course generation")
    else:
        print("Discharge Instructions generation")

    if bhc:
        df_discharge_train_segmented['input_bhc_rag'] = df_discharge_train_segmented[
            ['Chief Complaint', 'diagnoses', 'History of Present Illness']].apply(
            lambda x: '\n\n'.join(x), axis=1)
        df_discharge_test_phase_2_segmented['input_bhc_rag'] = df_discharge_test_phase_2_segmented[
            ['Chief Complaint', 'diagnoses', 'History of Present Illness']
        ].apply(lambda x: '\n\n'.join(x), axis=1)

        loader_train = DataFrameLoader(df_discharge_train_segmented, page_content_column='input_bhc_rag')

        loader_test = DataFrameLoader(df_discharge_test_phase_2_segmented, page_content_column='input_bhc_rag')

        df_document_train = loader_train.load()
        df_document_test = loader_test.load()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # check if the vectorstore exists
        if os.path.exists('bhc_vectorestore'):
            vectorstore = FAISS.load_local("bhc_vectorestore", embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_documents(df_document_train, embedding=embeddings)
            vectorstore.save_local('bhc_vectorestore')

    else:
        df_discharge_train_segmented['input_dis_rag'] = df_discharge_train_segmented[
            ['Chief Complaint', 'diagnoses', 'History of Present Illness', 'admission medications',
             'Discharge Medications',
             'Discharge Disposition', 'Discharge Diagnosis', 'Discharge Condition',
             ]].apply(
            lambda x: '\n\n'.join(x), axis=1)
        df_discharge_test_phase_2_segmented['input_dis_rag'] = df_discharge_test_phase_2_segmented[
            ['Chief Complaint', 'diagnoses', 'History of Present Illness', 'admission medications',
             'Discharge Medications',
             'Discharge Disposition', 'Discharge Diagnosis', 'Discharge Condition',
             ]].apply(lambda x: '\n\n'.join(x), axis=1)

        loader_train = DataFrameLoader(df_discharge_train_segmented, page_content_column='input_dis_rag')

        loader_test = DataFrameLoader(df_discharge_test_phase_2_segmented, page_content_column='input_dis_rag')

        df_document_train = loader_train.load()
        df_document_test = loader_test.load()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # check if the vectorstore exists
        if os.path.exists('di_vectorestore'):
            vectorstore = FAISS.load_local("di_vectorestore", embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = FAISS.from_documents(df_document_train, embedding=embeddings)
            vectorstore.save_local('di_vectorestore')

    retriever = vectorstore.as_retriever()

    process_items(start_idx, end_idx,bhc)
#