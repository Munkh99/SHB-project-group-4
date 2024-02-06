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

* install project libraries:
  ```
  pip install -r requirements.txt
  ```
* Data processing pipeline




1. **Install Dependencies:** Install the required libraries by running the following command in your project directory:

  ```
  pip install -r requirements.txt
  ```

2. **Data Preprocessing:**

Execute the preprocessing scripts in the following order:

- **Step 0: Tasks answers Preprocessing**

  Cleans and preprocess task answers data i.e. data about students. 
  ```
  python pre_processing/s0_task_answers.py -l pre_processing/log_s0.log -i data/raw/tasksanswers.parquet -o data/interim/
  ```

- **Step 1: Location Preprocessing**

  Finds students' home locations
  ```
  python pre_processing/s1_location.py -l pre_processing/log_s1.log -i data/raw/locationevent.parquet -o data/interim/
  ```

- **Step 2: Home location process**

  Cleans and preprocess time diaries.
  ```
  python pre_processing/s2_ground_truth_home.py -l pre_processing/log_s2.log -i data/interim/locationevent.csv -o data/interim/
  ```

- **Step 3: Time Diaries Preprocessing**

  Cleans and preprocess time diaries. 
  ```
  python pre_processing/s3_time_diaries.py -l pre_processing/log_s3.log -i data/raw/timediariesanswers.parquet -o data/interim/
  ```

- **Step 4: Matching Preprocessing**

  Match location coordinates with time diary answers. 
  ```
  python pre_processing/s4_matching.py -i_loc data/interim/locationevent.csv -i_td data/interim/time_diaries.csv -i_ta data/interim/tasksanswers.csv -o data/processed/
  ```

- **Step 5: Point of Interest Preprocessing**

  Cleans and preprocess POI file.
  ```
  python pre_processing/s4_point_of_interest.py -l pre_processing/log_s5.log -i data/raw/wu2013poi.json -o data/interim/
  ```

- **Step 6: POI Translator Preprocessing**

  Have to enumerate it in -c args: 'category', 'topics', 'timetable', 'seatingCapacity'. It needs to translate columns one by one due to the translator library limit.
  ```
  python pre_processing/s6_poi_translator.py -l pre_processing/log_s6.log -i data/interim/point_of_interest_non_trans.csv -i_t data/interim/translation.json -c 'category'
  ```
- **Step 7: Assign translation**
  In this step, categories, and topics are replaced with translated strings. We do not need all the categories. Save only relevant ones in the 'config.py' file in dictionary format.
   ```
  python pre_processing/s7_assign_translation.py -l pre_processing/log_s7.log -i data/interim/point_of_interest_non_trans.csv -i_t data/interim/translation.json -o data/processed/
  ```
  
- **Step 8: OSM Data scraping**
  
   This step is for obtaining relevant datasets from OSM Trentino dataset. It reads relevant file names from 'config.py' and then saves them in the 'data/formal' directory. Basically, automated way of choosing files from OSM.
  ```
  python pre_processing/s8_osm_select.py -l pre_processing/log_s8.log -i data/raw/point_of_interest_from_osm -o data/formal/
  ```
- **Step 9: University data scraping**
  
  This step is for obtaining additional university data from [Overpass Turbo](https://overpass-turbo.eu/). Data is downloaded with the following query and cleaned with the python script
  ```
    [out:json][timeout:180];
    (
        way["amenity"="university"](45.6621, 10.3706, 46.7811, 11.7512);
        relation["amenity"="university"](45.6621, 10.3706, 46.7811, 11.7512);
        node["amenity"="university"](45.6621, 10.3706, 46.7811, 11.7512);
    );
    out body;
    >;
    out skel qt;

  ```

  ```
  python pre_processing/s9_add_univeristy.py -l log_s9.log -i data/raw/poi_university.geojson -o data/processed
  ``'
  
Customize the script parameters according to your data and file paths.

## Directory Descriptions

- **data/raw:** Contains raw data files that need preprocessing.
- **data/processed:** Will store the processed data.
- **data/interim:** Keeps intermediate data files generated during processing.
- **data/formal:** Selected formal data files from OSM.
- **pre_processing:** Contains preprocessing scripts to clean and transform the raw data.
- **Code Libraries:**
- **ontology:** Contains ontologies or definitions for informal resources.





