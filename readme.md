# Repository for the Challenge Discharge Me!

More information about the challenge can be found on the [Discharge Me! Challenge FAQ page](https://stanford-aimi.github.io/discharge-me/#faq).

## Overview
This repository hosts the solution to the "Discharge Me!" challenge, which focuses on generating sections of discharge summaries from the MIMIC-IV dataset. The main objective is to automatically generate the "Brief Hospital Course" and "Discharge Instructions" sections. These components are critical for summarizing patient care and post-discharge actions succinctly, thereby assisting healthcare providers in delivering efficient and accurate patient care.

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
Generates missing sections in the discharge summaries. This script focuses on creating content for specific sections, depending on the needs identified within the discharge summary.

#### Parameters:
- `start_index`: The index of the first input to consider for generation.
- `end_index`: The total number of inputs to process for generation.
- `section_type`: Type of section to generate (`1` for a brief hospital course, `0` for discharge instructions).

## Usage
Ensure Python is installed along with the necessary dependencies, including:
langchain
pandas
ollama



Here's a simple example to run `generation.py`:

```bash
python generation.py 0 50 1
```
This it to generate the "Discharge Instruction" section for the first 50 discharge summaries of the test phase 2 dataset.

```bash
python generation.py 0 -1 0
```
This it to generate the "Brief Hospital Course" section for all discharge summaries of the test phase 2 dataset.
