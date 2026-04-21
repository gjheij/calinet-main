# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re

config = {
    "SOA": 7.5,
    "cs_duration": 8, 
    "us_duration": 0.5,
    "BIDS_Version": "1.1.0",
    "HED_Version": "8.2.0",
    "event_names": {"CSpr", "CSpu", "CSm", "USp", "USm", "USo"},
    "trim_window": [-10, 30],
    "blink_detection_settings": {
        "vel_thresh": 300.0,
        "min_fix_s": 0.06,
        "min_sacc_s": 0.01,
        "blink_threshold": 0.5,
        "max_blink_gap_samples": 2,
        "min_blink_s": 0.0,
        "mask_dropout": True,
        "dropout_threshold": 0.5,
        "pad_blink_before_s": 0.05,
        "pad_blink_after_s": 0.10,
    },
    # "smi_settings": {
    #     "vel_thresh": 300.0,
    #     "min_fix_s": 0.06,
    #     "min_sacc_s": 0.01,
    #     "blink_threshold": 0.5,
    #     "max_blink_gap_samples": 2,
    #     "min_blink_s": 0.02,
    # },
    "warning_at_mm": 9,
    "pupil_multiplication": {
        "AREA": 0.119,
        "DIAMETER": 0.00087743,
        "reference_distance": 700
    },
    "gap_bias": 0.6
}


stim_colors = {
    "CSpr": "#d62728",  # strong red – CS+ reinforced
    "CSpu": "#ff7f0e",  # orange – CS+ unreinforced
    "CSm": "#1f77b4",   # blue – CS-
    "USp": "#7f0000",   # dark red – actual aversive US
    "USm": "#17becf",   # teal – expected US during CS-
    "USo": "#7f7f7f",   # grey – omitted US during CS+
}


eyelink_regex = {
    "CAL_TYPE": re.compile(r"\[(.*?)\]"),
    "CAL_VALID": re.compile(r"ERROR\s+([\d.]+)\s+avg\.\s+([\d.]+)\s+max"),
    "ELCL_PROC": re.compile(r"ELCL_PROC\s+(\w+)"),
    "GAZE_COORDS": re.compile(
        r"GAZE_COORDS\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
    ),
    "PUPIL": re.compile(r"^PUPIL\s+(\w+)"),
    "RATE": re.compile(r"RATE\s+(\d+(?:\.\d+)?)"),
}

available_labs = {
    "amsterdam": {
        "Authors": [
            "Stemerding, Lotte",
            "Kindt, Merel"
        ],
        "MetaName": "Amsterdam",
        "Modalities": ["SCR", "ECG"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "ChannelRegex": {
            "scr_channel": "SCL",
            "ttl_channel": "marker",
            "ecg_channel": "ECG",
            "low_percentile": 60,   # filters out low US triggers
            "high_percentile": 95,  # filters out high 'start-of-block' triggers,
            "sampling_rate_hz": 1000.0  # default; verify with specified metadata
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": None
    },
    "austin": {
        "Authors": [
            "Cooper, Sam",
            "Dunsmoor, Joey"
        ],
        "MetaName": "Austin",
        "Modalities": ["SCR"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": [7, 8, 9, 10, 11, 12, 15, 18, 19, 20, 21, 25],
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "ChannelRegex": {
            "SCR": re.compile(r"EDA100C", re.I),
            "TTL": -1,
        },
        "gap_factor_between_acq_ext": 3, # shorter break between acq/ext
        "has_eyetrack": None
    },    
    "bielefeld": {
        "Authors": [
            "Ehlers, Mana",
            "Lonsdorf, Tina",
            "Cording, Conrad"
        ],
        "MetaName": "Bielefeld",
        "Modalities": ["SCR"],
        "Phenotype": {
            "Language": "german",
            "bfi": 60,
            "gad": 7,
            "ius": 18,
            "phq": 9,
            "soc": 12,
            "stai": 40
        },
        "ChannelRegex": {
            "SCR": re.compile(r"SCR - EDA100C", re.I),
            "TTL": re.compile(r"stim_marker", re.I),
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": None
    },
    "bologna": {
        "Authors": [
            "Battaglia, Simone"
        ],
        "MetaName": "Bologna",
        "Modalities": ["SCR", "ECG"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "ChannelRegex": {
            "SCR": re.compile(r"SCR", re.I),
            "ECG": re.compile(r"HR", re.I),
            "TTL": re.compile(r"CS trigger", re.I),
            "Shock": re.compile(r"Shock trigger", re.I)
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": None        
    },
    "bonn": {
        "Authors": [
            "Prygodda, Rebecca",
            "de Vries, Olivier",
            "Bach, Domink, R."
        ],
        "MetaName": "Bonn",
        "QuestionnaireLanguage": "german",
        "Modalities": ["SCR", "ECG", "RESP"],
        "Phenotype": {
            "Language": "german",
            "bfi": 60,
            "gad": 7,
            "ius": 18,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },        
        "ChannelRegex": {
            "SCR": re.compile(r"EDA100C", re.I),
            "ECG": re.compile(r"ECG100C", re.I),
            "RESP": re.compile(r"RSP100C", re.I),
            "TTL": re.compile(r"STP Input 0", re.I),
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": "asc"
    },
    "leuven": {
        "Authors": [
            "Chalkia, Anastasia",
            "Beckers, Tom"
        ],
        "MetaName": "Leuven",
        "Modalities": ["SCR", "ECG"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },        
        "ChannelRegex": {
            "SCR": re.compile(r"SCR", re.I),
            "ECG": re.compile(r"Heart Rate", re.I),
            "TTL": re.compile(r"CS MARKER", re.I),
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": None        
    },
    "london": {
        "Authors": [
            "Linnell, Josie",
            "Honarvar, Arshia",
            "Bach, Dominik, R."
        ],
        "MetaName": "London",
        "Modalities": ["SCR", "RESP", "ECG"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": "asc"        
    },    
    "newyork": {
        "Authors": [
            "Arpi, Evelyn",
            "Johnson, X."
        ],
        "MetaName": "New York",
        "Modalities": ["SCR"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "ChannelRegex": {
            "SCR": re.compile(r"EDA100C", re.I),
            "TTL": [
                re.compile(r"Stimulus", re.I),
                re.compile(r"Room", re.I),
            ]
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": None           
    },
    "reading": {
        "Authors": [
            "van Reekum, Carlien"
        ],
        "MetaName": "Reading",
        "Modalities": ["SCR"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },        
        "ChannelRegex": {
            "SCR": "SCR",
            "TTL": "TTL",
        },
        "gap_factor_between_acq_ext": 2, # shorter break between acq/ext
        "has_eyetrack": None       
    },
    "southampton": {
        "Authors": [
            "Morris, Jayne"
        ],
        "MetaName": "Southampton",
        "Modalities": ["SCR"],
        "Phenotype": {
            "Language": "english",
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },     
        "ChannelRegex": {
            "SCR": re.compile(r"GSR - GSR100C", re.I),
            "TTL": re.compile(r"STP Input 0", re.I),
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": None 
    },
    "stockholm": {
        "Authors": [
            "Lindhagen, Simon",
            "Olsson, X."
        ],
        "MetaName": "Stockholm",
        "Modalities": ["SCR"],
        "Phenotype": {
            "Language": "english",
            "items_already_corrected": True,
            "bfi": 30,
            "gad": 7,
            "ius": 12,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "ChannelRegex": {
            "SCR": re.compile(r"SCR", re.I),
            "ECG": re.compile(r"ECG", re.I),
            "TTL": re.compile(r"CS MARKER", re.I)
        },
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": "txt" 
    },
    "wuerzburg": {
        "Authors": [
            "Stegmann, Yannik",
            "Gamer, X."
        ],
        "MetaName": "Wuerzburg",
        "Modalities": ["SCR", "RESP", "ECG"],
        "Phenotype": {
            "Language": "german",
            "bfi": 30,
            "gad": 7,
            "ius": 18,
            "phq": 9,
            "soc": 12,
            "stai": 20
        },
        "ChannelRegex": ["SCR", "RESP", "ECG", "TTL"],
        "gap_factor_between_acq_ext": 5,
        "has_eyetrack": "asc"
    }
}

import_modules = {i: f"calinet.sites.{i}" for i in available_labs}