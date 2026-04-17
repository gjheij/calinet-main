# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import io
import os
import gzip
import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Union

import logging
logger = logging.getLogger(__name__)

def _json_converter(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def load_json(path: Union[Path, str]) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Union[Path, str], data: dict) -> None:
    path = Path(path)
    path.write_text(
        json.dumps(data, indent=4, default=_json_converter) + "\n",
        encoding="utf-8",
    )


def read_physio_tsv_headerless(path: Union[Path, str]) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        header=None,
        na_values=["n/a"],
        keep_default_na=True,
    )

    # read the columns
    json_path = path.with_suffix('').with_suffix('.json')
    if json_path.exists():
        meta_cols = load_json(json_path).get("Columns")
        logger.debug(f"Derived column names {meta_cols} from '{json_path}'")
        if len(meta_cols) == df.shape[1]:
            df.columns = meta_cols
    else:
        logger.debug(f"Could not derive column names; '{json_path}' does not exist. Current columns names: {list(df.columns)}")

    return df


def write_physio_tsv_gz_headerless(
        df: pd.DataFrame,
        out_path: Union[Path, str]
    ) -> None:

    """
    Always writes .tsv.gz with:
      - headerless
      - na_rep='n/a'
      - gzip mtime=0
      - empty filename in gzip header
    """
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"

    with open(tmp, "wb") as raw:
        with gzip.GzipFile(
            fileobj=raw,
            mode="wb",
            mtime=0,
            filename="",
        ) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as enc:
                df.to_csv(
                    enc,
                    sep="\t",
                    index=False,
                    header=False,
                    lineterminator="\n",
                    na_rep="n/a",
                )

    os.replace(tmp, out_path)
    
    # check if it actually exists
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Output file '{out_path}' not generated..")


def infer_json_sidecar(tsv_path: Path) -> Path:
    name = tsv_path.name
    if name.endswith(".tsv.gz"):
        return tsv_path.with_name(name.replace(".tsv.gz", ".json"))
    if name.endswith(".tsv"):
        return tsv_path.with_name(name.replace(".tsv", ".json"))
    raise ValueError("Input must be .tsv or .tsv.gz")    


def infer_output_tsv_gz(tsv_path: Path) -> Path:
    """
    Always return .tsv.gz path.
    """
    name = tsv_path.name
    if name.endswith(".tsv.gz"):
        return tsv_path
    if name.endswith(".tsv"):
        return tsv_path.with_name(name + ".gz")
    raise ValueError("Input must be .tsv or .tsv.gz")


def update_json_timestamp(meta: dict) -> None:
    if meta["Columns"][0] != "timestamp":
        meta["Columns"] = ["timestamp"] + meta["Columns"]

    meta.setdefault(
        "timestamp",
        {
            "LongName": "Time",
            "Description": "Time since start of recording",
            "Units": "s",
        },
    )


def reorder_columns(
        df: pd.DataFrame,
        meta: dict,
        desired: List[str]
    ) -> pd.DataFrame:

    current = meta["Columns"]

    if sorted(current) != sorted(desired):
        raise ValueError(
            f"--reorder must be permutation of JSON Columns.\n"
            f"Current: {current}\nRequested: {desired}"
        )

    idx = [current.index(name) for name in desired]
    df2 = df.iloc[:, idx].copy()
    meta["Columns"] = desired
    return df2


def convert_edf_to_asc(edf_file: str, asc_file: str) -> None:
    edf_path = Path(edf_file)
    asc_path = Path(asc_file)

    logger.info(f"Converting EDF file {edf_path} to ASC file {asc_path}")

    result = subprocess.run(
        ["edf2asc.exe", "-y", str(edf_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = result.stdout or ""

    if "Converted successfully" in stdout:
        logger.info(f"EDF successfully converted to {asc_path}")
        return

    logger.error(f"edf2asc failed for {edf_path}")
    logger.error(stdout)
    if result.stderr:
        logger.error(result.stderr)

    raise RuntimeError(f"Failed to convert EDF file: {edf_path}")


def convert_all_edfs_to_asc(
        raw_data_dir: str,
        overwrite: bool=False
    ) -> List:

    edf_files = []
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.lower().endswith(".edf"):
                edf_file = os.path.join(root, filename)
                edf_files.append(edf_file)

    asc_files = []
    if len(edf_files)>0:
        logger.info(f"{len(edf_files)} EDF files found")
        for edf_file in edf_files:
            asc_file = str(Path(edf_file).with_suffix(".asc"))
            if not os.path.exists(asc_file) or overwrite:
                try:
                    convert_edf_to_asc(
                        edf_file,
                        asc_file
                    )
                except Exception as e:
                    logger.error(f"Error converting {edf_file}: {e}")

            asc_files.append(asc_file)

        logger.info("EDF to ASC conversion complete.")

    return asc_files 

  
def find_smi_txt_files(
        raw_data_dir: str,
    ) -> List:

    smi_files = []
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.lower().startswith("eyetracking_") and filename.lower().endswith(".txt"):
                smi_file_path = os.path.join(root, filename)
                smi_files.append(smi_file_path)

    return smi_files
