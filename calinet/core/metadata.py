# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from calinet.core.io import (
    load_json,
    save_json
)

from calinet.core.units import normalize_bids_unit
from calinet.config import config, available_labs

from typing import Union

import logging
logger = logging.getLogger(__name__)

import calinet
metadata_file = os.path.join(
    calinet.PACKAGE_ROOT,
    'misc',
    'metadata.csv'
)

# read metadata file
df_meta = pd.read_csv(metadata_file)


def infer_modalities_from_hed(levels_dict: dict):
    """
    Infer CS and US modalities from an existing levels_dict
    (like the one you printed).

    Returns
    -------
    cs_modality : str
    us_modality : str
    """

    # Reverse maps
    cs_reverse_map = {
        "Visual-presentation": "visual",
        "Auditory-presentation": "auditory",
        "Somatosensory-stimulation": "somatosensory",
    }

    us_reverse_map = {
        "Auditory-presentation": "auditory",
        "Somatosensory-stimulation": "electrical", 
    }

    # Infer CS modality
    cs_hed = levels_dict.get("CSm", {}).get("HED", "")
    cs_modality = None
    for hed_tag, mod_name in cs_reverse_map.items():
        if hed_tag in cs_hed:
            cs_modality = mod_name
            break

    if cs_modality is None:
        raise ValueError("Could not infer CS modality from HED.")

    # Infer US modality
    us_hed = (
        levels_dict.get("USp", {}).get("HED", "") or
        levels_dict.get("USm", {}).get("HED", "")
    )

    us_modality = None
    for hed_tag, mod_name in us_reverse_map.items():
        if hed_tag in us_hed:
            us_modality = mod_name
            break

    if us_modality is None:
        raise ValueError(f"Could not infer US modality from HED levels: {levels_dict.keys()}.")

    return cs_modality, us_modality


def build_hed_map(cs_modality: str, us_modality: str) -> dict:
    cs_mod = cs_modality.lower()
    us_mod = us_modality.lower()

    cs_mod_map = {
        "visual": "Visual-presentation",
        "auditory": "Auditory-presentation",
        "somatosensory": "Somatosensory-stimulation",
    }

    us_mod_map = {
        "auditory": "Auditory-presentation",
        "electrical": "Somatosensory-stimulation",
    }

    if cs_mod not in cs_mod_map:
        raise ValueError("CS modality must be visual, auditory, or somatosensory")

    if us_mod not in us_mod_map:
        raise ValueError("US modality must be auditory or electrical")

    cs_tag = cs_mod_map[cs_mod]
    us_tag = us_mod_map[us_mod]

    hed_map = {
        "CSpr": f"Sensory-event,{cs_tag},Experimental-condition/CS-plus,Experimental-condition/Reinforced",
        "CSpu": f"Sensory-event,{cs_tag},Experimental-condition/CS-plus,Experimental-condition/Unreinforced",
        "CSm":  f"Sensory-event,{cs_tag},Experimental-condition/CS-minus",
        "USp":  f"Sensory-event,{us_tag},Aversive-stimulus",
        "USm":  f"Experimental-condition/CS-minus,Experimental-condition/Expectation-violation,{us_tag}",
        "USo":  f"Experimental-condition/CS-plus,Experimental-condition/Unreinforced,Experimental-condition/Expectation-violation"
    }

    return hed_map


def map_participants_tsv(
        input: Union[pd.DataFrame, str],
        filename=None
    ) -> pd.DataFrame:
    """
    Clean and map participants.tsv to valid BIDS format.
    - Renames 'gender' to 'sex'
    - Maps common gender values to BIDS-compliant values
    """
    
    if not isinstance(input, pd.DataFrame):
        if isinstance(input, str):
            input = Path(input)
        
        df = pd.read_csv(input, sep="\t")
    else:
        df = input.copy()

    # Rename gender -> sex (BIDS preferred)
    if "gender" in df.columns:
        df = df.rename(columns={"gender": "sex"})

    if "sex" not in df.columns:
        raise ValueError("No 'sex' or 'gender' column found.")

    # Mapping dictionary
    sex_mapper = {
        "male": "M",
        "Male": "M",
        "m": "M",
        "M": "M",
        "1": "M",

        "female": "F",
        "Female": "F",
        "f": "F",
        "w": "F",
        "F": "F",
        "2": "F",

        "other": "O",
        "o": "O",
        "nonbinary": "O",

        "na": "n/a",
        "n/a": "n/a",
        "nan": "n/a",
        "": "n/a"
    }

    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(sex_mapper)
    )

    if filename is not None:
        df.to_csv(filename, sep="\t", index=False, na_rep="n/a")

    # check if json exists
    if isinstance(filename, (str, Path)):
        if isinstance(filename, str):
            filename = Path(filename)

        json_path = filename.with_suffix(".json")
        if json_path.exists():

            # load json
            metadata = load_json(json_path)

            # update metadata
            metadata["sex"] = {
                "Description": "Self-reported biological sex of the participant.",
                "Levels": {
                    "M": "Male",
                    "F": "Female",
                    "O": "Other",
                    "n/a": "Not available"
                }
            }

            # remove gender if present
            if "gender" in metadata:
                del metadata["gender"]

            save_json(json_path, metadata)
            
    return df


def create_dataset_description(
        bids_root: str,
        lab_name:str=None
    ) -> None:
    """
    Create a BIDS-compliant dataset_description.json for a consolidated
    pupillometry dataset following BEP020/BEP045.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to the root folder of the final BIDS dataset.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    modalities = list(available_labs[lab_name]["Modalities"])
    trim_window = config.get("trim_window", [-10, 30])

    if available_labs[lab_name].get("has_eyetrack") is not None:
        modalities.append("eyetracking")

    dataset_description = {
        "Name": f"{available_labs.get(lab_name).get('MetaName')} CALINET Fear-Conditioning Dataset",
        "BIDSVersion": str(config.get("BIDS_Version")),
        "DatasetType": "raw",

        # lab-specific modalities
        "Modalities": modalities,

        # BIDS extensions relevant to eye tracking / pupillometry
        "BIDSVersionExtension": ["BEP020", "BEP045"],

        "Authors": available_labs.get(lab_name).get("Authors"),

        "Description": (
            f"Lab-specific CALINET fear-conditioning dataset contributed as part of the "
            f"CALINET2 consortium. The dataset contains physiology (at least electrodermal activity) and, where "
            f"available, eye-tracking recordings acquired during a differential fear-"
            f"conditioning experiment with geometric visual cues and electrical "
            f"stimulation as the unconditioned stimulus."
        ),

        "Acknowledgements": (
            "This dataset was collected within the CALINET/CALINET2 research framework. "
            "The BIDS conversion was performed with the CALINET Converter. The dataset "
            "structure reflects fear-conditioning phases and preprocessing choices used "
            "for harmonized downstream analyses."
        ),

        "HowToAcknowledge": (
            "Please cite the original lab-specific dataset, the CALINET2 consortium "
            "paper(s), and the BIDS conversion workflow when using these data."
        ),

        "GeneratedBy": [
            {
                "Name": "CALINET Converter",
                "Version": "1.0.0",
                "CodeURL": "https://github.com/gjheij/calinet-main",
                "Description": (
                    "Conversion to BIDS with harmonized event alignment and metadata "
                    "standardization across CALINET labs."
                )
            }
        ],

        "License": "CC-BY-4.0",
        "GeneratedDate": str(date.today()),
        "HEDVersion": config.get("HED_Version"),

        # project-specific metadata fields
        "StudyType": "Differential fear conditioning",
        "Consortium": "CALINET2",
        "InstitutionName": {available_labs.get(lab_name).get("MetaName")},

        "TaskDescription": (
            "The experiment consisted of two blocks corresponding to fear-acquisition "
            "and fear-extinction phases. In the BIDS organization, these are represented "
            "as separate tasks: 'task-acquisition' and 'task-extinction'. Participants "
            "viewed a fixation cross on a gray background while geometric figures were "
            "presented on screen, and electrical stimulations were delivered during the "
            "experiment."
        ),

        "Instructions": (
            "Participants were instructed that the experiment would begin with a fixation "
            "cross on a gray background, followed by images of geometric figures. They "
            "were asked to look attentively at the screen throughout the experiment. "
            "Participants were also informed that they would receive electrical "
            "stimulations, and that there may or may not be a relationship between the "
            "geometric figures and the electrical stimulations. The experiment was "
            "divided into two blocks."
        ),

        "TaskPhases": [
            {
                "Name": "acquisition",
                "BIDSLabel": "task-acquisition",
                "Description": "Fear-acquisition phase of the differential conditioning experiment."
            },
            {
                "Name": "extinction",
                "BIDSLabel": "task-extinction",
                "Description": "Fear-extinction phase of the differential conditioning experiment."
            }
        ],

        "PreprocessingDescription": (
            f"All recordings were trimmed to a window spanning {trim_window[0]} seconds before the "
            f"first event marker to {trim_window[1]} seconds after the last event marker. This window "
            "was used to standardize the analyzable time range across recordings."
        ),

        "SamplingAlignment": (
            "Where eye-tracking data were available, eye-tracking time series were "
            "realigned to physiology recordings using the first shared event marker as "
            "temporal anchor."
        ),

        "EyeTrackingAlignmentApplied": available_labs.get(lab_name).get("has_eyetrack") is not None,

        "Notes": (
            "The dataset follows a fear-conditioning design consistent with acquisition "
            "and extinction phases used in calibration-oriented human fear-conditioning "
            "research. The uploaded reference paper describes a differential fear-"
            "conditioning framework with high vs. low US expectation, acquisition and "
            "extinction phases, simple geometric cues, electrical stimulation, and pupil "
            "size as one of the suggested observables."
        )
    }

    output_file = bids_root / "dataset_description.json"
    save_json(output_file, dataset_description)
    logger.info(f"Created: {output_file}")


def get_modalities(lab_cfg):
    modalities = set(lab_cfg.get("Modalities", []))

    if lab_cfg.get("has_eyetrack") is not None:
        modalities.add("eyetracking")

    return sorted(modalities)


def create_readme(bids_root: str, lab_name:str):
    """
    Create a human-readable README file for the consolidated
    BIDS pupillometry dataset.
    
    Parameters
    ----------
    bids_root : str or Path
        Path to the root folder of the final BIDS dataset.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    time_window = config.get("time_window", [-10, 30])

    readme_text = f"""
BIDS CALINET Dataset ({available_labs.get(lab_name).get("MetaName")})
====================================================================

Overview
--------
This dataset contains psychophysiological recordings collected during a 
differential fear-conditioning experiment as part of the CALINET/CALINET2 consortium.

Available modalities:
{", ".join(get_modalities(available_labs.get(lab_name)))}

Participants viewed geometric stimuli while receiving electrical stimulation.
The experiment is divided into two phases:
- Acquisition (task-acquisition)
- Extinction (task-extinction)

Standard
--------
The dataset conforms to:
- BIDS Version {config.get("BIDS_Version")}
- BEP020 (Eye Tracking Extension, if applicable)
- BEP045 (Physiology Extension)

Data Structure
--------------
- Physiological and eye-tracking recordings are stored in:
  /sub-*/physio/
- Task phases are encoded as:
  - task-acquisition
  - task-extinction
- Events files follow BIDS conventions (*_events.tsv)
- Metadata are provided via JSON sidecars (*_physio.json)
- Participant-level data:
  - participants.tsv
  - participants.json
- Phenotype data:
  /phenotype/

Preprocessing
-------------
The following preprocessing steps were applied consistently across all subjects:

- All recordings were trimmed to a window of:
  {time_window} seconds relative to the first and last event marker
- Event markers were used to align recordings across modalities
- If eye-tracking data were available, they were realigned to physiology
  using the first event marker as temporal reference

Notes
-----
- Not all subjects contain eye-tracking data
- Missing modalities are expected depending on site-specific acquisition
- Data were harmonized across labs to support comparability

License
-------
This dataset is distributed under CC-BY-4.0.

Acknowledgement
---------------
Please cite:
- The original lab-specific dataset
- The CALINET/CALINET2 consortium
- The BIDS conversion pipeline (CALINET Converter)

Generated On
------------
{date.today()}

    """

    output_file = bids_root / "README"
    output_file.write_text(readme_text.strip() + "\n", encoding="utf-8")

    logger.info(f"Created: {output_file}")    


def create_bidsignore(
    bids_root,
    patterns=None,
    overwrite=False
):
    """
    Create or update a .bidsignore file in a BIDS dataset root.

    Parameters
    ----------
    bids_root : str or Path
        Path to the root of the BIDS dataset.
    patterns : list of str
        File names or glob patterns to ignore.
        Example: ["pupilbench_corpus_overview.csv", "*.log"]
    overwrite : bool
        If True, overwrite existing .bidsignore.
        If False, append new patterns only.
    """

    if patterns is None:
        patterns = []

    bids_root = Path(bids_root)
    bidsignore_path = bids_root / ".bidsignore"

    # Normalize patterns (remove leading slashes)
    patterns = [p.lstrip("/") for p in patterns]

    if overwrite or not bidsignore_path.exists():
        existing_patterns = set()
    else:
        with open(bidsignore_path, "r") as f:
            existing_patterns = set(
                line.strip() for line in f.readlines() if line.strip()
            )

    updated_patterns = existing_patterns.union(patterns)

    with open(bidsignore_path, "w") as f:
        for pattern in sorted(updated_patterns):
            f.write(pattern + "\n")

    logger.info(f"Created: {bidsignore_path}")


def _parse_pair_mm(s: str):
    """Parse '(a, b) mm' -> [a, b] floats."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    m = re.match(r"^\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*mm\s*$", s, flags=re.I)
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2))]


def _parse_resolution_px(s: str):
    """Parse 'WxH px' or '(W, H)' -> [W, H] ints."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()

    m = re.match(r"^\s*(\d{3,5})\s*[xX]\s*(\d{3,5})\s*px\s*$", s, flags=re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]

    m = re.match(r"^\(\s*(\d{3,5})\s*,\s*(\d{3,5})\s*\)\s*$", s)
    if m:
        return [int(m.group(1)), int(m.group(2))]

    return None


def _parse_refresh_rate_hz(s: str):
    """Parse '60 Hz' or '56-76 Hz' -> int (prefer the first number)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _parse_triple_mm(s: str):
    """Parse '(x, y, z) mm' or 'x,y,z' -> [x,y,z] floats."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()

    m = re.match(
        r"^\(\s*([\-0-9]+(?:\.[0-9]+)?)\s*,\s*([\-0-9]+(?:\.[0-9]+)?)\s*,\s*([\-0-9]+(?:\.[0-9]+)?)\s*\)\s*mm\s*$",
        s, flags=re.I
    )
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))]

    m = re.match(
        r"^\s*([\-0-9]+(?:\.[0-9]+)?)\s*,\s*([\-0-9]+(?:\.[0-9]+)?)\s*,\s*([\-0-9]+(?:\.[0-9]+)?)\s*$",
        s
    )
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))]

    return None


def _first_scalar(series: pd.Series):
    """Return first non-null scalar in a Series, else None."""
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    val = s.iloc[0]
    return None if (isinstance(val, float) and np.isnan(val)) else val


def stimulus_presentation_from_metadata(df: pd.DataFrame, lab: str):
    """
    Build a BIDS-like StimulusPresentation dict for a given lab column from your metadata table.

    Expected metadata rows (Parameter):
      - Screen Size
      - Screen Resolution
      - Screen Refresh Rate
      - Eyetracker Eye-Screen Distance (X, Y, Z)   (optional, used for ScreenDistance)
        OR Screen Distance / Viewing Distance (if you have those)

    df: wide table with columns like ['Data','Parameter', 'Wuerzburg', 'Stockholm', ...]
    lab: column name for the lab (must match df column)
    """
    if lab not in df.columns:
        raise KeyError(f"Lab '{lab}' not found in metadata columns: {list(df.columns)}")

    # Convenience lookup: rows by Parameter (and optionally Data)
    def cell(parameter, data=None):
        mask = (df["Parameter"] == parameter)
        if data is not None and "Data" in df.columns:
            mask &= (df["Data"] == data)
        return _first_scalar(df.loc[mask, lab])

    screen_size = _parse_pair_mm(cell("Screen Size", data="Screen") or cell("Screen Size"))
    screen_res  = _parse_resolution_px(cell("Screen Resolution", data="Screen") or cell("Screen Resolution"))
    refresh_hz  = _parse_refresh_rate_hz(cell("Screen Refresh Rate", data="Screen") or cell("Screen Refresh Rate"))

    # ScreenDistance (mm): prefer eye-screen distance Z, else a direct distance field if present
    dist_xyz = _parse_triple_mm(cell("Eyetracker Eye-Screen Distance (X, Y, Z)", data="Eye")
                               or cell("Eyetracker Eye-Screen Distance (X, Y, Z)"))
    screen_distance = int(round(dist_xyz[2])) if dist_xyz else None

    # Fallbacks if you store distance directly
    if screen_distance is None:
        direct = cell("Screen Distance", data="Screen") or cell("Viewing Distance") or cell("Screen Distance")
        if direct is not None:
            # accept "700", "700 mm", "0.7 m"
            s = str(direct).strip()
            m_mm = re.match(r"^(\d+(?:\.\d+)?)\s*mm$", s, flags=re.I)
            m_m  = re.match(r"^(\d+(?:\.\d+)?)\s*m$", s, flags=re.I)
            m_raw = re.match(r"^(\d+(?:\.\d+)?)$", s)
            if m_mm:
                screen_distance = round(float(m_mm.group(1)))
            elif m_m:
                screen_distance = round(float(m_m.group(1)) * 1000.0)
            elif m_raw:
                screen_distance = round(float(m_raw.group(1)))

    stimulus = {
        "StimulusPresentation": {
            "ScreenDistance": screen_distance,          # mm (int), if available
            "ScreenOrigin": ["top", "left"],            # common convention;
            "ScreenRefreshRate": refresh_hz,            # Hz (int)
            "ScreenResolution": screen_res,             # [W, H] pixels
            "ScreenSize": screen_size,                  # [W, H] mm
        }
    }

    # Optional: drop None fields to keep JSON clean
    stimulus["StimulusPresentation"] = {
        k: v for k, v in stimulus["StimulusPresentation"].items() if v is not None
    }

    return stimulus


def _meta_for(df_meta: pd.DataFrame, modality_name: str) -> pd.DataFrame:
    return df_meta.loc[df_meta["Data"].eq(modality_name.upper())].set_index("Parameter")


def _get(meta: pd.DataFrame, param: str, lab_name: str, default=None):
    
    # data must exist
    assert lab_name in list(meta.columns), f"Metadata for lab '{lab_name}' is missing"

    # safer access with default if missing
    try:
        val = meta.at[param, lab_name]
    except KeyError:
        return default
    return val


def fill_general(lab_name: str, modality_name: str, json_content: dict) -> dict:
    meta = _meta_for(df_meta, modality_name)

    json_content["Manufacturer"] = _get(meta, "Manufacturer", lab_name)
    json_content["ManufacturersModelName"] = _get(meta, "Manufacturer Model", lab_name)
    json_content["DeviceSerialNumber"] = _get(meta, "Manufacturer Device Serial Number", lab_name)
    json_content["SoftwareVersion"] = _get(meta, "Software Version", lab_name)

    # we should have SamplingFrequency by now; default to df_meta
    if not "SamplingFrequency" in json_content:
        json_content["SamplingFrequency"] = None
    
    sr = json_content.get("SamplingFrequency")
    if not isinstance(sr, (int, float)):
        try:
            sr = float(_get(meta, "Sampling Rate", lab_name))
        except Exception:
            raise ValueError(f"Could not derive SamplingFrequency. Specify 'Sampling Rate' -> '{lab_name}' -> '{modality_name}' -> 'Sampling Rate'") 
        
        logger.warning(f"Fetched SamplingFrequency from metadata: {sr} [this is a last-ditch effort.. Reading from physio-file is safer!]")
        json_content["SamplingFrequency"] = sr

    return json_content


def fill_scr_json(lab_name: str, json_content: dict) -> dict:

    meta = _meta_for(df_meta, "SCR")
    scr = json_content.setdefault("scr", {})
    scr["SCRCouplerType"] = _get(meta, "EDA coupler type", lab_name)
    scr["SCRCouplerVoltage"] = _get(meta, "EDA coupler voltage", lab_name)
    scr["Placement"] = _get(meta, "Placement", lab_name)

    # fill adhoc
    if scr["Units"] is None:
        unit = normalize_bids_unit(_get(meta, "units", lab_name))
        logger.warning(f"Filling 'Units' from metadata: {unit}")
        scr["Units"] = unit

    return json_content


def fill_ecg_json(lab_name: str, json_content: dict) -> dict:
    
    meta = _meta_for(df_meta, "ECG")
    ecg = json_content.setdefault("ecg", {})
    ecg["Placement"] = _get(meta, "Placement", lab_name)
    
    # fill adhoc
    if ecg["Units"] is None:
        unit = normalize_bids_unit(_get(meta, "units", lab_name))
        logger.warning(f"Filling 'Units' from metadata: {unit}")
        ecg["Units"] = unit

    return json_content


def fill_ppg_json(lab_name, json_content):

    meta = _meta_for(df_meta, "PPG")
    ppg = json_content.setdefault("ppg", {})
    ppg["SensorType"] = _get(meta, "Sensor Type", lab_name)
    ppg["Placement"] = _get(meta, "Placement", lab_name)

    # fill adhoc
    if ppg["Units"] is None:
        unit = normalize_bids_unit(_get(meta, "units", lab_name))
        logger.warning(f"Filling 'Units' from metadata: {unit}")
        ppg["Units"] = unit

    return json_content


def fill_resp_json(lab_name, json_content):

    meta = _meta_for(df_meta, "RESP")
    resp = json_content.setdefault("resp", {})
    resp["SensorType"] = _get(meta, "Sensor Type", lab_name)
    resp["Placement"] = _get(meta, "Placement", lab_name)

    # fill adhoc
    if resp["Units"] is None:
        unit = normalize_bids_unit(_get(meta, "units", lab_name))
        logger.warning(f"Filling 'Units' from metadata: {unit}")
        resp["Units"] = unit

    return json_content
