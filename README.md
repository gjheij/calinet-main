# CALINET-Management

Project Management for CALINET Data Processing Team

## Converters Overview
This converter processes physiological data files, converting them into ``JSON`` and ``TSV`` formats according to predefined rules, effectively transforming the data into the BIDS (Brain Imaging Data Structure) format.
The code is modular and organized into multiple Python scripts for better maintainability.

## File Structure
- ``main.py``: The main script that orchestrates the processing and conversion tasks.
- ``data_conversion.py``: Functions for converting data to JSON and TSV formats.
- ``file_utils.py``: Functions for directory creation and file structure handling.
- ``templates.py``: Contains JSON templates.
- ``logger.py``: Sets up the logging configuration.

## Installation
1) Clone the Repository

```bash
git clone https://gitlab.com/bachlab/repositories/CALINET-Data-Management
```

## How to Run
1) Install Required Dependencies

```bash
pip install numpy pandas bioread scipy
```

2) Navigate to the Converter Directory

```bash
cd Converters/Lab_Specific_Code/Bonn
```

3) Run the Main Script

```bash
python main.py --base_folder <path_to_raw_data> --output_folder <path_to_converted_data>
```

Replace <path_to_raw_data> with the path to your raw data directory.
Replace <path_to_converted_data> with the desired output directory.

## Default Behavior

If you do not provide ``--base_folder`` or ``--output_folder``, the script defaults to using `'Raw Data'` and `'Converted Data'` directories in the script's directory.

You can also simply create the `'Raw Data'` and `'Converted Data'` folder as well in specific converter folder and place your files which you want to convert in 'Raw Data' folder. 

The script will process the data files and generate the specified JSON and TSV files in the output directory according to the BIDS format.
