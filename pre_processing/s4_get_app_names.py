import json
import time

import pandas as pd
from utils import PATH_TO_INTERIM_DATA, PATH_TO_RAW_DATA, get_logger, PATH_TO_PROCESSED_DATA, ROOT_DIR, \
    centermost_point, radius_of_gyration, get_total_distance_covered, _intervalindex_to_columns
import argparse
import os
from config import application_grouping, app_not_found_suffix
import requests
from bs4 import BeautifulSoup


def find_category(word, categories):
    for category, words in categories.items():
        if word in words:
            return category
    print(f"{word} is not in the group")
    return 'not_found'


def find_category_group(word, categories):
    try:
        for category_group, suffixes in categories.items():
            if any(suffix in word.lower() for suffix in suffixes):
                return category_group

        # print(f"{word} is not in any group")
        return 'other'
    except(Exception) as e:
        print(e)
        return 'other'


def main(path_to_sensor, output_path):
    logger.info("started")
    start = time.time()
    logger.info("reading sensor data")
    sensor = pd.read_csv(path_to_sensor)

    # s = sensor[sensor['userid'] < 20]
    # s.to_csv(os.path.join(ROOT_DIR, 'data', 'raw', 'applicationevent.csv'), index=False)

    application_names = sensor.applicationname.unique()
    app_category_from_file = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'appcategories.csv'))
    app_category_from_file = app_category_from_file.set_index('app_id')['category'].to_dict()

    category_db = {}
    app_not_found = set()
    for (i, app) in enumerate(application_names):
        if i % 100 == 1:
            print(f'{i}/{len(application_names)}')
        # for the first time run, it searches app names from web, from 2nd it goes to this part
        if app == 'com.gotokeep.yoga.intl':
            print()

        if app in app_category_from_file.keys():
            if app_category_from_file[app] != 'other':
                category = app_category_from_file[app]
                category_db[app] = category
            else:
                # TODO find other way to get category for app_not_found files, or its possible to do by hand in file
                print(app)
                k = find_category_group(app, app_not_found_suffix)
                category_db[app] = k
        else:
            url = f'https://play.google.com/store/apps/details?id={app}'
            try:
                response = requests.get(url)
                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    # Parse the HTML content of the page
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Find the <script> tag with type 'application/ld+json'
                    script_tag = soup.find('script', {'type': 'application/ld+json'})

                    # Extract the text content of the <script> tag
                    script_content = script_tag.text.strip()

                    # Parse the JSON content
                    json_data = json.loads(script_content)

                    # Access the value of 'applicationCategory'
                    category = json_data.get('applicationCategory', None)

                    category_group = find_category(category.lower(), application_grouping)
                    category_db[app] = category_group
                    # print(f'{app} - {category.lower()} - {category_group}')
                else:
                    # print(f'Failed to retrieve information for {app}. Status code: {response.status_code}')
                    ##for debug
                    if app not in app_not_found:
                        app_not_found.add(app)
                        print(app)
                    ##
                    category_db[app] = 'other'
            except (AttributeError, json.JSONDecodeError, KeyError) as e:
                print("Error:", e)
                category_db[app] = 'other'
                print(f'{app} - not_found')
    category_db = pd.DataFrame(list(category_db.items()), columns=['app_id', 'category'])
    category_db.to_csv(os.path.join(ROOT_DIR, 'data', 'appcategories.csv'), index=False)
    logger.info(f"finished in: {round(time.time() - start)} [s]'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s4.log')
    parser.add_argument('-i', '--input', help='path to timediaries',
                        default=os.path.join(PATH_TO_RAW_DATA, 'applicationevent.csv'))

    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_INTERIM_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)

    application_categories_path = os.path.join(ROOT_DIR, 'data', 'appcategories.csv')
    if os.path.exists(application_categories_path):
        app_cat_exists = True
    else:
        with open(application_categories_path, 'w') as file:
            file.write('app_id,category\n')
        app_cat_exists = False
    logger.info(f"app category exists : {app_cat_exists}")
    main(args.input, args.output)
