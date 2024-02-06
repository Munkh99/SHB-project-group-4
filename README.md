# SHB-project-group-4
for Studies on Human Behavior course project

# Informal resources processing

Structure for processing informal resources.

## Project Structure

```plaintext
project_root/
│
├── data/
│   ├── raw/
│   │   └── # Raw data files
│   ├── processed/
│   │   └── # Processed data files
│   └── interim/
│       └── # Intermediate data files
├── plots/
├── pre-processing/
│   ├── config.py
│   ├── s1_time_diary.py
│   ├── s2_demographic.py 
│   ├── s3_matching.py
│   ├── s4_get_app_names.py
│   ├── s5_features.py
│   ├── s6_join_features.py
│   ├── s7_mood_normalization.py
│   └── utils.py
├── analyzie_viz.py
├── main.py
├── requirements.txt # Project libraries
│
```

## Getting Started

Follow these steps to set up and run the project:

# install project libraries:
  ```
  pip install -r requirements.txt
  ```
# Data processing pipeline

Download required datasets in /data/raw/ directory. 

Execute the preprocessing scripts in the following order:

- **Step 1: Timediary dataset processing**

  ```
  python pre_processing/s1_timediary.py -l pre_processing/log_s1.log -i data/raw/td_ita.dta -o data/interim/
  ```

- **Step 2: Demographic dataset processing**

  ```
  python pre_processing/s2_demographic.py -l pre_processing/log_s2.log -i data/raw/data4diarynew_ITA.dta -o data/interim/
  ```

- **Step 3: Matching**

  Matches outputs of step 1 and 2. 
  ```
  python pre_processing/s3_matching.py -l pre_processing/log_s3.log -i data/interim/demographic.csv -i2 data/interim/timediary.csv -o data/interim/
  ```

- **Step 4: Get app names**

  Scrapes Google Play Store to get application group names. 
  ```
  python pre_processing/s4_get_app_names.py -l pre_processing/log_s3.log -i data/raw/applicationevent.csv -o data/interim/
  ```

- **Step 5: Process features**

  Process features include notification sensor and application sensor. 
  ```
  python pre_processing/s5_features.py -l pre_processing/log_s5.log -i data/raw/notificationevent.csv -o data/interim/

  python pre_processing/s5_features.py -l pre_processing/log_s5.log -i data/raw/applicationevent.csv -o data/interim/
  ```

- **Step 6: Join features**
  from /raw/interim folder, it reads files '*_feature.csv'
  ```
  python pre_processing/s6_join_features.py -l pre_processing/log_s6.log -i data/interim/  -o data/processed/
  ```
  
- **Step 7: mood normalization**

   ```
  python pre_processing/s7_mood_normalization.py -l pre_processing/log_s7.log -i data/processed/final_data_encoded.csv -o data/processed/
  ```

You can customize the script parameters according to your data and file paths.

## Directory Descriptions

- **data/raw:** Contains raw data files that need preprocessing.
- **data/processed:** Will store the processed data.
- **data/interim:** Keeps intermediate data files generated during processing.
- **data/formal:** Selected formal data files from OSM.
- **pre_processing:** Contains preprocessing scripts to clean and transform the raw data.
- **Code Libraries:**
- **ontology:** Contains ontologies or definitions for informal resources.





