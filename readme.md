# Repository for the Challenge Discharge Me!

More information about the challenge can be found on the [Discharge Me! Challenge FAQ page](https://stanford-aimi.github.io/discharge-me/#faq).

## Overview
This repository hosts the solution to the "Discharge Me!" challenge, which focuses on generating sections of discharge summaries from the MIMIC-IV dataset. The main objective is automatically generating the "Brief Hospital Course" and "Discharge Instructions" sections.

## Scripts

### `aggregate_discharge.py`
This script augments the discharge summary table by integrating other relevant tables from the MIMIC-IV dataset, providing a fuller context and background for each patient’s stay.

#### Features:
- Reads additional tables to augment the data.
- Integrates multiple data sources to enhance the discharge summary.

### `discharge_dataset.py`
Processes the discharge summaries by segmenting them into specific sections, truncating excessive content, and aggregating the results into a structured table.

#### Features:
- Segments discharge summaries into manageable parts.
- Truncates sections that are overly verbose.
- Aggregates sections to form a structured output table.

### `generation.py`
Generates missing sections in the discharge summaries based on RAG and LLama3 (using Ollama to serve the llama3 model). Two prompt templates are curated for this purpose, and RAG retrieves a target section's word count as the target output's word count.

#### Parameters:
- `start_index`: The index of the first input to consider for a generation.
- `end_index`: The total number of inputs to process for generation.
- `section_type`: Type of section to generate (`1` for a brief hospital course, `0` for discharge instructions).

## Usage
Ensure Python is installed along with the necessary dependencies, including:
- langchain
- pandas
- ollama

The file path used in those files should be adapted to your file path.


First, run the `aggregate_discharge.py` script to augment the discharge summary table. 
Then, run the `discharge_dataset.py` script to process the discharge summaries. 
Finally, run the `generation.py` script to generate the missing sections in the discharge summaries.
The Jupyter notebook `discharge_me_analysis.ipynb` can be used to predict the word count of the missing sections, which is not the preferred way in the final paper.



Here's a simple example to run `generation.py`:

```bash
python generation.py 0 50 1
```
This generates the "Discharge Instruction" section for the first 50 discharge summaries of the test phase 2 dataset.

```bash
python generation.py 0 -1 0
```
This generates the "Brief Hospital Course" section for all test phase 2 dataset discharge summaries.
