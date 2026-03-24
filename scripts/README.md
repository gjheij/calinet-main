# CALINET Data Sharing Pipeline

This repository provides a standardized workflow for converting, selecting, anonymizing (blinding), and transforming CALINET datasets for cross-lab sharing and downstream analysis.

The pipeline consists of five main steps:

1. **Data Conversion (`calinet_convert`)**
2. **Data Selection (`calinet_select`)**
3. **Data Blinding (`calinet_blinder`)**
4. **Conversion to Autonomate (`calinet_autonomate`)**
5. **Conversion to EzySCR (`calinet_ezyscr`)**

---

## Overall Data Flow

```
sourcedata/   →   converted/   →   exports/
(raw lab data)    (BIDS-like)      (subset)
                                   ├→ blinded/     (anonymized)
                                   ├→ autonomate/  (analysis-ready)
                                   └→ ezyscr/      (analysis-ready)
```

---

# 1️⃣ Data Conversion (`calinet_convert`)

Converts lab-specific raw datasets into a standardized BIDS-like structure.

### Purpose

* Harmonize heterogeneous lab formats
* Create consistent folder structure across sites
* Generate logs for traceability

### Example

```bash
python -m calinet_convert `
    --input-dir <some_path>/sourcedata/lab `
    --output-dir <some_path>/converted/lab `
    --n-workers 4 `
    --clean `
    --debug
```

### Behavior

* Automatically detects subjects and files in raw dataset
* Applies lab-specific conversion logic (`convert_data`)
* Supports multiprocessing (`--n-workers`)
* Generates logs:

  * `log.log` (final)
  * intermediate logs cleaned automatically

### Output

```bash
converted/
  bielefeld/
    participants.tsv
    participants.json
    phenotype/
    sub-*
```

If `--output-dir` is not provided, it is inferred by replacing `sourcedata` with `converted` in the input path 

---

# 2️⃣ Data Selection (`calinet_select`)

Selects a subset of participants across sites while controlling for site bias.

### Features

* Stratified sampling by site
* Avoids re-exporting subjects
* Reproducible via random seed
* Tracks export history
* Reusable randomly selected participants across software packages

### Example

```bash
python calinet_select.py `
    --input-dir '<some_path>\converted'
    --n 10 `
    --recipient-lab Test
```

which will print something like this:
```log
[2026-03-24 15:56:38.512] [-] [INFO] calinet.select - Log-file: <some_path>\derivatives\exports\log.log
[2026-03-24 15:56:38.512] [-] [INFO] calinet.select - Output directory: <some_path>\derivatives\exports
[2026-03-24 15:56:38.512] [-] [INFO] calinet.select - Sampling method: equal
[2026-03-24 15:56:38.518] [-] [INFO] calinet.select - Requested N: 10
[2026-03-24 15:56:38.518] [-] [INFO] calinet.select - Recipient lab: Test
[2026-03-24 15:56:38.551] [-] [INFO] calinet.select - Loaded existing registry: <some_path>\derivatives\exports\participants_registry.tsv
[2026-03-24 15:56:38.601] [-] [INFO] calinet.select - Saved registry to: <some_path>\derivatives\exports\participants_registry.tsv
[2026-03-24 15:56:38.602] [-] [INFO] calinet.select - Batch ID: EXP_2026-03-22_Test_N10_seed1234
```

The `<some_path>\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234` directory can be directly used for the conversion to data formats required by other software packages (e.g., `blinding` stimulus events, `EzySCR`, and `Autonomate`).

To select a new subset of non-overlapping subjects, pass the ``--participants-tsv`` flag pointing to the ``registry``-file:

```bash
python .\calinet_select.py `
  --input-dir ..\..\..\converted\ `
  --n 150 `
  --recipient-lab Test2 `
  --participants-tsv ..\..\..\derivatives\exports\participants_registry.tsv
```

which will output something like:
```log
[2026-03-24 15:56:38.512] [-] [INFO] calinet.select - Log-file: <some_path>\derivatives\exports\log.log
[2026-03-24 15:56:38.512] [-] [INFO] calinet.select - Output directory: <some_path>\derivatives\exports
[2026-03-24 15:56:38.512] [-] [INFO] calinet.select - Sampling method: equal
[2026-03-24 15:56:38.518] [-] [INFO] calinet.select - Requested N: 150
[2026-03-24 15:56:38.518] [-] [INFO] calinet.select - Recipient lab: Test2
[2026-03-24 15:56:38.551] [-] [INFO] calinet.select - Loaded existing registry: <some_path>\derivatives\exports\participants_registry.tsv
[2026-03-24 15:56:38.601] [-] [INFO] calinet.select - Saved registry to: <some_path>\derivatives\exports\participants_registry.tsv
[2026-03-24 15:56:38.602] [-] [INFO] calinet.select - Batch ID: EXP_2026-03-24_Test2_N150_seed1234
Traceback (most recent call last):
  File "<some_path>\code\calinet-main\scripts\calinet_select.py", line 134, in <module>
    selected_df = select_subjects(
  File "\\caian-nas\Exchange\CALINET2\code\calinet-main\calinet\exports\selector.py", line 594, in select_subjects
    raise ValueError(
ValueError: Requested n=150, but only 98 eligible subjects are available
```

It will throw an error if you select more participants than available after filtering for previously-exported participants.
To include participants that already have been exported before, use ``--include-previously-exported``.

### Output

```bash
exports/
  EXP_<date>_<lab>_N<...>_seed<...>/
    participants.tsv
    participants.json
    metadata.json
```

---

# 3️⃣ Data Blinding (`calinet_blinder`)

Copies only selected subjects and removes identifying information.

### Features

* Accepts export output (`participants.tsv`)
* Flattens multi-site datasets into a single subject-level structure
* Optional modality filtering
* Skips site-specific phenotype data (`--skip-pheno`)
* Blinds event files (`events.tsv`, `.json`)

### Example

```bash
# insert output from calinet_select.py (see above)
python calinet_blinder.py `
  --subjects-tsv '..\..\..\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234\participants.tsv' `
  --input-dir '..\..\..\converted\' `
  --output-dir '..\..\..\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234\blinded' `
  --task acquisition `
  --modalities scr
```

### Output

```bash
exports/
  EXP_2026-03-22_Test_N10_seed1234/
    blinded
      sub-*
      log.log
      metadata.json
      participants.json
      participants.tsv
```


All subjects are placed at the root:

```bash
sub-CalinetAustin01/
sub-CalinetBielefeld03/
```

(no site folders)

---

# 4️⃣ Conversion to Autonomate (`calinet_autonomate`)

Converts the blinded dataset into Autonomate-compatible format.

### Purpose

* Prepare data for automated processing pipelines
* Standardize structure expected by Autonomate

### Example

```bash
# the selector will create a consortium-wide participant.tsv file, with all subjects
# we can use this to convert all ~400 datasets in one go
python calinet_autonomate.py `
  --subjects-tsv '..\..\..\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234\participants.tsv' `
  --input-dir '..\..\..\converted\' `
  --output-dir '..\..\..\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234\autonomate' `
  --task acquisition
```

### Output

```bash
exports/
  EXP_2026-03-22_Test_N10_seed1234/
    autonomate
      sub-*
      log.log
      metadata.json
      participants.json
      participants.tsv
```

---

# 5️⃣ Conversion to EzySCR (`calinet_ezyscr`)

Converts the dataset into EzySCR format for SCR analysis.

### Purpose

* Prepare physiological data for skin conductance analysis
* Ensure compatibility with EzySCR pipelines

### Example

```bash
python calinet_ezyscr.py `
  --subjects-tsv '..\..\..\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234\participants.tsv' `
  --input-dir '..\..\..\converted\' `
  --output-dir '..\..\..\derivatives\exports\EXP_2026-03-22_Test_N10_seed1234\EzySCR' `
  --task acquisition
```

### Output

```bash
exports/ 
  EXP_2026-03-22_Test_N10_seed1234/
    EzySCR
      sub-*
      log.log
      metadata.json
      participants.json
      participants.tsv
```

# Prepare Calibench upload

Calibench expects compressed files that it then later unzips into it's own data folder. 
We can quickly zip everything with `calinet_separate.py`:


---

# Design Principles

* **Reproducibility**

  * Fixed random seeds
  * Logged exports and conversions

* **Bias control**

  * Equal or proportional sampling across sites

* **Modularity**

  * Each step is independent and composable

* **BIDS-aligned**

  * Uses `participants.tsv` + `participants.json`

* **Traceability**

  * Logs created at every stage

---

# Notes

* Blinded datasets are not full BIDS datasets unless additional metadata files are added.
* Phenotype data are excluded during blinding due to cross-site heterogeneity.
* Subject IDs should be globally unique (recommended: include site name).

---

# Future Extensions

* Multi-variable stratified sampling (e.g. site + sex + age)
* Automatic regeneration of `participants.tsv` after blinding
* Full BIDS export packaging
* QC pipeline integration

---

# Contact

For questions or contributions, please contact the CALINET data team (jurjen.heij@uni-bonn.de).
