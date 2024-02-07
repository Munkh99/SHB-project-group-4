import glob
import time
import numpy as np
import pandas as pd
from utils import PATH_TO_INTERIM_DATA, PATH_TO_RAW_DATA, get_logger, PATH_TO_PROCESSED_DATA, ROOT_DIR, parse_interval, \
    _intervalindex_to_columns

import argparse
import os
from config import sensors_config


def determine_day_period(hour) -> str:
    if 6 <= hour < 10:
        return "morning"
    if 10 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    else:
        return "night"


def get_features(path_to_sensors) -> pd.DataFrame:
    features_union = None
    for sensor_file_name in path_to_sensors:
        logger.info(f'load {os.path.basename(sensor_file_name)}')
        sensor_name = os.path.splitext(os.path.basename(sensor_file_name))[0].split('_')[0]

        sensors = pd.read_csv(
            sensor_file_name,
            parse_dates=['start_interval', 'end_interval']
        )

        if sensor_name not in sensors_config:  ##added line in config file, big sensors are commented, need to ask
            continue

        # Attempt to get the list of columns to exclude
        columns_to_exclude = sensors_config.get(sensor_name, {}).get('columns_to_exclude', [])

        # Filter out columns that exist in the DataFrame
        columns_to_exclude_existing = [col for col in columns_to_exclude if col in sensors.columns]

        # Drop the columns that exist in the DataFrame
        sensors.drop(columns=columns_to_exclude_existing, inplace=True)

        types_dict = {'userid': int, 'experimentid': str}
        for col, col_type in types_dict.items():
            sensors[col] = sensors[col].astype(col_type)
        sensors['interval'] = parse_interval(sensors)
        sensors.drop(['start_interval', 'end_interval'], axis=1, inplace=True)
        sensors.set_index(['userid', 'experimentid', 'interval'], inplace=True)

        if features_union is None:
            features_union = sensors
        else:
            features_union = features_union.join(sensors, how='outer', validate="one_to_one")

    # assert all(features_union[['latitude', 'longitude']].nona())
    # features_union = features_union.fillna(0)
    features_union = features_union.dropna() #also drops user timediary datas

    intervals = pd.Series(features_union.index.get_level_values(2))
    features_union['hour'] = intervals.apply(lambda x: x.mid.hour).values
    # source http://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    features_union['sin_hour'] = np.sin(2 * np.pi * features_union.hour / 24)
    features_union['cos_hour'] = np.cos(2 * np.pi * features_union.hour / 24)
    features_union['day_period'] = features_union.hour.apply(determine_day_period)
    features_union = pd.get_dummies(features_union, columns=['day_period'], drop_first=True)

    features_union = features_union.reset_index(names=['userid', 'experimentid', 'interval'])
    _intervalindex_to_columns(features_union)
    return features_union


def cut_dataset(X):
    user_duration = X.groupby('userid')['start_interval'].agg(['min', 'max'])
    # Calculate the duration by subtracting the minimum timestamp from the maximum timestamp
    user_duration['duration'] = user_duration['max'] - user_duration['min']
    # Convert the duration to days
    user_duration['duration_days'] = user_duration['duration'].dt.days
    # days of users are almost same cuz after merging features, it drops nans
    logger.info("Duration days info:")
    logger.info(user_duration['duration_days'].describe())  # mean is 22

    selected_users = user_duration[user_duration['duration_days'] >= 22]

    selected_data = pd.DataFrame()
    for user, row in selected_users.iterrows():
        user_data = X[(X['userid'] == user) & (X['start_interval'] <= row['min'] + pd.Timedelta(days=22))]
        selected_data = selected_data._append(user_data)
    return selected_data


def main(path_to_sensors, output_path):
    logger.info("started")
    start = time.time()
    file_pattern = os.path.join(path_to_sensors, '*_feature.csv')

    file_list = glob.glob(file_pattern)
    X = get_features(file_list)

    X_22 = cut_dataset(X)

    X_22.to_csv(os.path.join(output_path, f'final_data.csv'), index=False)

    X_22_encoded = pd.get_dummies(X_22, columns=['gender', 'nationality', 'department', 'cohort', 'degree',
                                                    'location', 'first2w', 'week', 'what', 'where',
                                                    'withw', 'uniproblem'
                                                    ], dtype=int)

    X_22_encoded.to_csv(os.path.join(output_path, f'final_data_encoded.csv'), index=False)


    ### insight
    logger.info(f'Number of samples: {len(X_22)}')
    logger.info(f'Number of userd: {len(X_22.userid.unique())}')
    logger.info(f'Average sample for each user: {round(len(X_22) / len(X_22.userid.unique()), 2)}')
    k = X_22.groupby("userid").size()

    logger.info(f'Average sample for each user: {round(k.mean(), 2)}')
    logger.info(f'Std sample for each user: {round(k.std(), 2)}')
    logger.info(f"finished in: {round(time.time() - start)} [s]'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s6.log')
    parser.add_argument('-i', '--input', help='path to sensors',
                        default=os.path.join(PATH_TO_INTERIM_DATA))
    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_PROCESSED_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.output)
