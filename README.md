# Boosting Neurodegenerative Disorder Screenings with Machine Learning 

This repository contains the code for the tool described in 'todo'.

---
## Overview
<!-- [Introduction](#introduction)-->
- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [2.1 Data Pre-Processing](#step-21-data-pre-processing)
  - [2.1 RBD Prediction](#step-22-rbd-prediction)
---
## 1. Installation
### Step 1.1:  Create a Python 3.9 Environment
- Option A: **Using Conda (recommended)**
```
conda create -n aktiRBD python=3.9
conda activate aktiRBD
```
- Option B: **Using a Virtual Environment**
```
python3.9 -m venv aktiRBD
# On macOS/Linux:
source aktiRBD/bin/activate
# On Windows:
aktiRBD\Scripts\activate
```

### Step 1.2: Clone the Repository
Create a new folder for your experiments and download the codebase. 
```
mkdir aktiRBD_experiment
cd aktiRBD_experiment 
git clone git@github.com:db-2010/aktiRBD.git 
cd aktiRBD
```

### Step 1.3: Install the Package
Simply run `python install.py` to 
1. check if OpenMP, a non-python multiprocessing dependency of XGBoost, is available and otherwise try to install it.
2. Install the Python Package and dependencies from `pyproject.toml` using `pip`.

after this , the output should look like
```
... 
2025-03-10 17:11:22 [INFO] Verifying package installation...
2025-03-10 17:11:22 [INFO] Quality-control check passed: aktiRBD imported successfully!
2025-03-10 17:11:22 [INFO] Installation successful!
```

---
## 2. Usage
The code is modular and has two entry points: i) `aktiRBD-process` as a standalone tool for actigraphy data processing 
and ii) `aktiRBD-analysis` for RBD prediction using the extracted nocturnal motion features. 
### Step 2.0: File Organization
To run the code, you need the raw actigraphy files and a `metadata.csv` file. 
Supported formats for actigraphy binaries are
- **Axivity AX6:** `.cwa`
- **GENEActiv:** `.bin`
- **Actiwatch:** `.bin` (todo)
- **Unspecific:** `.csv` (todo)

The `metadata.csv` file should have one row per subject/record and contain at least the columns 
- ***filename:*** the filename of the actigraphy data.
- ***ID:*** a random identifier for the subject the data is taken from.
- ***diagnosis:*** a string indicating the cohort group of the subject, 
e.g. 'RBD' for RBD diagnosed subject and 'HC' for neg.
- ***train/test:*** should be set to 'test' for external data.
- ***exclude:*** can be used to exclude certain subject from the analysis by setting it to '1', otherwise empty.

| #   | filename               | ID     | diagnosis | train/test | exclude |
|-----|------------------------|--------|-----------|------------|---------|
| 1   | \<name1>.<cwa/bin/csv> | <ID-1> | RBD       | test       |         |
| 2   | \<name2>.<cwa/bin/csv> | <ID-2> | RBD       | test       | 1       |
| 3   | \<name3>.<cwa/bin/csv> | <ID-3> | HC        | test       |         |
| ... | ...                    | ...    | ...       | ...        | ...     |

**Note:** if the user is only interested in the preprocessing part of the pipeline, the columns ***diagnosis*** and 
***train/test*** are irrelevant and can be left empty. 

The default organization of the files is shown below. The binaries are simply listed under `./data/raw/` and the 
metafile should be stored at `./data/raw/meta/metadata.csv`. 
If different file locations are necessary, see [Step 2.1: Data Pre-Processing](#step-21-data-pre-processing).
```
aktiRBD_experiment/ (the main folder of your experiment)  
  ├── aktiRBD (the cloned repo)
  │     ├── README.md 
  │     ├── src/ 
  │     ├── ... 
  ├── data/ 
  │     ├── raw/ (put your actipgraphy binaries here...)
  │     │    ├── <name1>.<cwa/bin/csv> 
  │     │    ├── <name2>.<cwa/bin/csv>
  │     │    ├── ...
  │     │    ├── meta/  (...and don't foget the metadata file.)
  │     │    │     ├── metadata.csv
  │     ├── processed/ (this is where the results of processing will be stored by default)
  │     │    ├── <name2>/
  │     │    ├── ...
  │    ...  ...
  ├── results/ (this is where results of the classifer will be stored by default)
 ... 
```

### Step 2.1: Data Pre-Processing
This step reads in the binary data, performs preprocessing (uniform resampling, Butterworth bandpass filtering, 
auto-calibration, non-wear detection, and sleep segmentation) and computes the nocturnal motion features for classification.
If the files are located at the default positions as described above, you can simply run
```bash 
aktiRBD-preprocess 
```
with the default options. To see all available options, run `aktiRBD-preprocess --help`. The most important ones are

File organization:
 - **`-d, --data_dir`** <br> 
 **Description:** (str) The directory (relative to `root_dir`) that contains the raw actigraphy files. <br> 
 **Default:** `./data/raw/` <br> 
 **Note:** Only needed if binary files should be located outside the default directory. 
- **`-m, --meta_file`** <br> 
 **Description:** (str) The path to the metadata CSV file, relative to `root_dir`. <br> 
 **Default:** `./data/raw/meta/metadata.csv`<br>
- **`-o, --out_dir`** <br>
**Description:** (str) The directory where processed data and features will be stored.<br>
**Default:** ./data/processed/ (relative to `root_dir`)<br>
**Note:** By default, both the processed data and calculated features are stored. Use the next option to save only
features if disk space is a concern.<br>

Operational Flags:
- **`-ns, --no_store`** <br>
**Description:** When provided, the script saves only the calculated features and not the processed data.<br>
**Default:** False. <br>
**Note:** If flag is not activated,  the processed actigraphy data is stored as `.parquet` file. 
(~2GB for a 7d recording at 100Hz.)
- **`-cp, --create_plots`** <br>
**Description:** Whether to create plots of the raw and processed actigraphy data.<br>
**Default:** True. <br>
- **`-sf, --skip_feature_calc`** <br>
**Description:**  If enabled, will skip the calculation of motion features.<br>
**Default:** False. <br>
**Note:** Use this if you're only interested in the processing part.

  
**Note:** If you are only interested in the processing part and want flexibility concerning parameters like bandpass 
frequencies etc., you can also provide a modified config file (`-c`, see `./config/preprocessing.yaml`).


---

### Step 2.2. RBD Prediction
To produce RBD predictions based on the extracted nocturnal motion features and the pretrained models, you can run
```
aktiRBD-analysis -d <path>/<to>/<processed>/<data> -m <path>/<to>/<meta>/<file>
```
If you did not change the default paths in [Step 2.1: Data Pre-Processing](#step-21-data-pre-processing), you can omit the 
extra options and simply call `aktiRBD-analysis`. For more details about available options, run
`aktiRBD-analysis -h`. 
The results of the analysis will be stored under `./results/pipeline/run_<date_id>/` and consist of a `.csv` file 
containing individual predictions and a `.json` file containing classification metrics. 

---

### Note on Reproducibility:
The code is seeded, fully deterministic and reproducible. 
However, there are some small platform dependencies linked to the RNG in XGBoost that lead to (small) variations
if executed on different OS like Linux or macOS [
[1](https://github.com/dmlc/xgboost/issues/310),
[2](https://github.com/dmlc/xgboost/issues/3046),
[3](https://github.com/dmlc/xgboost/issues/3523),
[4](https://github.com/dmlc/xgboost/pull/3781),
[5](https://github.com/dmlc/xgboost/issues/9584),
[6](https://discuss.xgboost.ai/t/colsample-by-tree-leads-to-not-reproducible-model-across-machines-mac-os-windows/1709),
[7](https://discuss.xgboost.ai/t/colsample-bytree-seemingly-not-random/106/20)
].
The only way to mitigate this is to match the C++ compilers used in both OS',
e.g. by building XGBoost from source using GCC on macOS:
```
conda activate aktiRBD_v2
brew install gcc@11  
git clone https://github.com/dmlc/xgboost.git
cd xgboost
git checkout 'v2.0.3'   
git submodule update --init --recursive 
mkdir build
cd build
CC=gcc-11 CXX=g++-11 cmake ..     
make -j12
cd ../python_package
pip install . 
```
--- 
# Cite Us

### todo 
