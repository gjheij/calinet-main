import sys
import os
import logging
import argparse

from conversion_utils.templates import (
    DATASET_DESCRIPTION_TEMPLATE,
    README_CONTENT,
    EVENTS_JSON_TEMPLATE,
)


def convert_data(raw_data_dir: str, converted_dataset_dir: str, dataset_name: str):
    """
    Process files in all subfolders for data conversion and organization.
    This version aggregates phenotype and shock ratings data into common files.
    """
    logging.info(f"Processing files in {raw_data_dir}")

    sts = clean_output_directory(converted_dataset_dir)
    if not sts:
        print(
            "Output directory has some content already. Please clear it to use the converter."
        )
        return

    # Create dataset_description.json and readme.txt
    create_dataset_descriptors(
        converted_dataset_dir,
        dataset_name,
        DATASET_DESCRIPTION_TEMPLATE,
        README_CONTENT,
    )
    print("---------------------------")

    # --- Global conversion step: Convert all EDF files to ASC files ---
    convert_all_edfs_to_asc(raw_data_dir) 
    print("---------------------------")

    # Gather info for all participants
    print("Gathering participant info...")
    (all_participant_info_tsv, all_participant_pheno_tsv) = handle_participant_info(
        raw_data_dir, converted_dataset_dir
    )

    print("---------------------------")
    # Gather all subject folders - folders starting with prefix "sub"
    subject_dirs = find_sub_dirs(raw_data_dir)

    print(f"\n{len(subject_dirs)} subjects found in dataset.")
    print("---------------------------")

    # Create a common phenotype folder in the output root
    phenotype_common_dir = os.path.join(converted_dataset_dir, "phenotype")
    os.makedirs(phenotype_common_dir, exist_ok=True)

    # Process aggregated phenotype (questionnaire) data for all subjects at once.
    handle_pheno(all_participant_pheno_tsv, phenotype_common_dir)

    # Initialize aggregated shock ratings dictionary.
    aggregated_shock_ratings = {}

    # ------------------------ Process each subject ------------------------
    for subject_raw_data_path in subject_dirs:
        subject_folder_name = os.path.basename(subject_raw_data_path)
        subject_name = extract_subject_name(subject_folder_name)
        print(f"\nProcessing: {subject_name}")

        # create sessions
        subject_converted_data_dir = os.path.join(converted_dataset_dir, subject_name)
        create_sessions_dirs(subject_converted_data_dir)

        # eyetracking
        handle_eyetracking(
            subject_raw_data_path,
            subject_converted_data_dir,
            subject_name
        )

        # other
        event_onsets = handle_physio(
            subject_raw_data_path,
            subject_converted_data_dir,
            subject_name
        )

        # events
        task_ratings = handle_events(
            subject_raw_data_path,
            subject_converted_data_dir,
            subject_name,
            event_onsets,
            EVENTS_JSON_TEMPLATE,
        )

        # Shock ratings: accumulate data for aggregated files.
        accumulate_shock_ratings(aggregated_shock_ratings, subject_name, task_ratings)
        create_beh_json(subject_converted_data_dir, subject_name)

    # Write aggregated shock ratings to the common phenotype folder.
    write_aggregated_shock_ratings(aggregated_shock_ratings, phenotype_common_dir)

    print("\n---------------------------\n")

    print("Anonymizing...")
    lab_name = dataset_name.split(" ")[-1]
    anonymize_converted_data(converted_dataset_dir, lab_name)
    print("Anonymization complete.")
    print("\n---------------------------\n")

    print("Dataset conversion complete.")


if __name__ == "__main__":
    dataset_name = "CALINET Bonn"

    parser = argparse.ArgumentParser(description="Arguments for the data converter.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Raw Data",
        help="Input directory for raw data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Converted Data",
        help="Output directory to save converted data",
    )
    args = parser.parse_args()

    raw_data_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logging (this writes detailed logs to log.txt and suppresses console output)
    setup_logging(output_dir)

    print(f"Raw dataset: {raw_data_dir}")
    print(f"Saving converted dataset to: {output_dir}\n")

    logging.info(f"Created base output directory {output_dir}")
    convert_data(raw_data_dir, output_dir, dataset_name)
