# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from calinet.templates.english import *
from calinet.templates.german import *


# questionnaire metadata mapping
QUESTIONNAIRE_DICT = {
    "german": {
        "bfi": {
            30: BFI30_SPEC_DE,
            60: BFI60_SPEC_DE
        },
        "gad": {
            7: GAD7_SPEC_DE
        },
        "ius": {
            18: IUS18_SPEC_DE
        },
        "phq": {
            9: PHQ9_SPEC_DE
        },
        "soc": {
            12: SOC12_SPEC_DE
        },
        "stai": {
            20: STAI20_SPEC_DE,
            40: STAI40_SPEC_DE
        },
        "ratings": {
            "pre-acq": PRE_ACQUISITION_RATINGS_DE,
            "post-acq": POST_ACQUISITION_RATINGS_DE,
            "post-ext": POST_EXTINCTION_RATINGS_DE
        }
    },
    "english": {
        "bfi": {
            30: BFI30_SPEC_EN,
        },
        "gad": {
            7: GAD7_SPEC_EN
        },
        "ius": {
            12: IUS12_SPEC_EN
        },
        "phq": {
            9: PHQ9_SPEC_EN
        },
        "soc": {
            12: SOC12_SPEC_EN
        },
        "stai": {
            20: STAI20_SPEC_EN
        },
        "ratings": {
            "pre-acq": PRE_ACQUISITION_RATINGS_EN,
            "post-acq": POST_ACQUISITION_RATINGS_EN,
            "post-ext": POST_EXTINCTION_RATINGS_EN
        }
    }
}


def get_questionnaire_spec(language: str, questionnaire: str, items):
    """
    Retrieve questionnaire metadata from QUESTIONNAIRE_DICT.

    Parameters
    ----------
    language : str
        Language key (e.g., "German", "English").
    questionnaire : str
        Questionnaire name (e.g., "BFI", "GAD", "Ratings").
    items : int or str
        Number of items (e.g., 30) or rating stage (e.g., "pre-acq").

    Returns
    -------
    dict
        The questionnaire specification.

    Raises
    ------
    KeyError
        If the requested combination does not exist.
    """

    try:
        return QUESTIONNAIRE_DICT[language.lower()][questionnaire.lower()][items]
    except KeyError as e:
        raise KeyError(
            f"Invalid questionnaire lookup: "
            f"language={language}, questionnaire={questionnaire}, items={items}"
        ) from e
    

DATASET_DESCRIPTION_TEMPLATE = {
    "Name": "Lab name",
    "BIDSVersion": "1.1.0"
}

# these are place holders; should be filled in by metadata from labs
README_CONTENT = "Dataset generated using the Bachlab Calinet Data Converter.\n"

PARTICIPANTS_JSON_TEMPLATE = {
    "Description": "Participant demographic information for the fear conditioning dataset.",
    "participant_id": {
        "Description": "The identification of the participant."
    },
    "age": {
        "Description": "The age of the participant defined as an integer.",
        "Units": "years",
    },
    "sex": {
        "Description": "Self-reported biological sex of the participant.",
        "Levels": {
            "M": "Male",
            "F": "Female",
            "O": "Other",
            "n/a": "Not available"
        }
    }
}


EVENTS_JSON_TEMPLATE = {
    "onset": {
        "LongName": "Onset",
        "Description": "Stimulus onset",
        "Units": "s"
    },
    "duration": {
        "LongName": "Duration",
        "Description": "Stimulus duration",
        "Units": "s",
    },
    "event_type": {
        "LongName": "Event Type",
        "Description": "Event type as defined in abstract terms",
        "Levels": {
            "CS+": "Conditioned stimulus coupled with US",
            "CS-": "Conditioned stimulus never coupled with US",
            "US": "Unconditioned stimulus",
            "block_start": "Start of trial block",
            "block_end": "End of trial block",
        },
    },
    "stimulus_name": {
        "LongName": "Identifier",
        "Description": "Stimulus identifier",
        "Levels": {
            "diamond": "A diamond with \u2026 degrees visual angle",
            "square": "A square with \u2026 degrees visual angle",
            "shock": "An electric shock",
            "block_start": "Start of trial block",
            "block_end": "End of trial block",
        },
    },
    "task_name": {
        "LongName": "Name of the task",
        "Description": "Name of the task",
        "Levels": {
            "habituation": "Fear habituation task",
            "acquisition": "Fear acquisition task",
            "extinction": "Fear extinction task",
        },
    }
}


EYE_JSON_TEMPLATE = {
    "Description": "Eye-tracking recording (gaze position and pupil size) collected during the fear conditioning task",
    "Columns": [
        "timestamp",
        "x_coordinate",
        "y_coordinate",
        "pupil_size"
    ],
    "PhysioType": "eyetrack",
    "StartTime": 0,
    "SampleCoordinateSystem": "gaze-on-screen",
    "timestamp": {
        "LongName": "Time",
        "Description": "a continuously increasing identifier of the sampling time registered by the device",
        "Origin": "System startup",
        "Units": "s"
    },    
    "x_coordinate": {
        "LongName": "Gaze position (x)",
        "Description": "Gaze position x-coordinate of the recorded eye",
        "Units": "mm",
    },
    "y_coordinate": {
        "LongName": "Gaze position (y)",
        "Description": "Gaze position y-coordinate of the recorded eye",
        "Units": "mm",
    },
    "pupil_size": {
        "Description": "Pupil diameter",
        "Units": "mm"
    }
}


ECG_JSON_CONTENT = {
    "Columns": [
        "timestamp",
        "ecg"
    ],
    "Manufacturer": "Biopac Systems",
    "ManufacturersModelName": "ECG100C",
    "DeviceSerialNumber": "1711008598",
    "SoftwareVersion": "Biopac AcqKnowledge 5.0.2",
    "StartTime": 0,
    "PhysioType": "generic",
    "timestamp": {
        "LongName": "Time",
        "Description": "a continuously increasing identifier of the sampling time registered by the device",
        "Origin": "System startup",
        "Units": "s"
    },      
    "ecg": {
        "Description": "ECG Recording",
        "Placement": "underneath the right clavicle, as well as the left and right costal margin",
        "Units": "mV",
    }
}


RESP_JSON_CONTENT = {
    "Columns": [
        "timestamp",
        "resp"
    ],
    "Manufacturer": "<Manufacturer-Name>",
    "ManufacturersModelName": "<Manufacturer-Model-Name>",
    "DeviceSerialNumber": "<Device-Serial-Number>",
    "SoftwareVersion": "<Software-version>",
    "StartTime": 0,
    "PhysioType": "generic",
    "timestamp": {
        "LongName": "Time",
        "Description": "a continuously increasing identifier of the sampling time registered by the device",
        "Origin": "System startup",
        "Units": "s"
    },    
    "resp": {
        "Description": "Respiratory Recording",
        "SensorType": "Belt",
        "Placement": "Chest",
        "Units": "V"
    }
}


PPG_JSON_CONTENT = {
    "Columns": [
        "timestamp",
        "ppg"
    ],
    "Manufacturer": "Biopac Systems",
    "ManufacturersModelName": "PPG100C",
    "DeviceSerialNumber": "",
    "SoftwareVersion": "Biopac AcqKnowledge 5.0.2",
    "StartTime": 0,
    "PhysioType": "generic",
    "timestamp": {
        "LongName": "Time",
        "Description": "a continuously increasing identifier of the sampling time registered by the device",
        "Origin": "System startup",
        "Units": "s"
    },      
    "ppg": {
        "SensorType": "Optoelectronic Sensor",
        "Placement": "Index and Middlefinger",
        "Units": "V",
    }
}


SCR_JSON_CONTENT = {
    "Columns": [
        "timestamp",
        "scr"
    ],
    "Manufacturer": "Biopac Systems",
    "ManufacturersModelName": "EDA100C/MEC100C",
    "DeviceSerialNumber": "1711001172",
    "SoftwareVersion": "Biopac AcqKnowledge 5.0.2",
    "StartTime": 0,
    "PhysioType": "generic",
    "timestamp": {
        "LongName": "Time",
        "Description": "a continuously increasing identifier of the sampling time registered by the device",
        "Origin": "System startup",
        "Units": "s"
    },      
    "scr": {
        "Description": "SCR Recording",
        "SCRCouplerType": "",
        "SCRCouplerVoltage": "",
        "Placement": "thenar and hypothenar eminences of the palmar surface",
        "Units": "uS",
        "MeasureType": "EDA-total"
    }
}


EYE_PHYSIO_EVENTS_JSON_TEMPLATE = {
    "Columns": [
        "onset",
        "duration",
        "trial_type",
        "blink",
        "message"
    ],
    "Description": "Messages logged by the measurement device",
    "OnsetSource": "timestamp",
    "blink": {
      "Description": "Gives status of the eye.",
      "Levels": {
          "0": "Indicates if the eye was open.",
          "1": "Indicates if the eye was closed."
      }
    },
    "message": {
      "Description": "String messages logged by the eye-tracker."
    },
    "trial_type": {
      "Description": "Event type as identified by the eye-tracker's model.",
      "Levels": {
          "fixation": "Indicates a fixation.",
          "saccade": "Indicates a saccade."
      }
    }
}


PARTICIPANT_INFO_SPEC = {
    "MeasurementToolMetadata": {
        "Description": "sub_info",
        "TermURL": "https://osf.io/k34rf/",
    },
    "participant_id": {"Description": "Unique identifier for each participant"},
    "recorded_at": {"Description": "Date and Time"},
    "room_temperature": {
        "Description": "Room Temperature (in degree Celsius)",
        "Units": "oC",
    },
    "humidity": {"Description": "Relative Humidity (in percent)", "Units": "a.u."},
    "sex": {
        "Description": "Self-reported biological sex of the participant.",
        "Levels": {
            "M": "Male",
            "F": "Female",
            "O": "Other",
            "n/a": "Not available"
        }
    },
    "age": {"Description": "Age of the participant", "Units": "year"},
    "handedness": {
        "Description": "Participant handedness",
        "Levels": {
            "left": "Left-handed",
            "right": "Right-handed",
            "both": "Ambidextrous/Both",
        }
    }
}
