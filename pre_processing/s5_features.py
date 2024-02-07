import time
import scipy
import pandas as pd
from utils import PATH_TO_INTERIM_DATA, PATH_TO_RAW_DATA, get_logger, PATH_TO_PROCESSED_DATA, ROOT_DIR, \
    centermost_point, radius_of_gyration, get_total_distance_covered, _intervalindex_to_columns
import argparse
import os


def compute_windows_intervals(contribution: pd.DataFrame, window_size_mins: int) -> pd.IntervalIndex:
    timestamp = 'timestamp'
    start_interval = contribution[timestamp].dt.floor('S') - pd.Timedelta(minutes=int(window_size_mins / 2))
    end_interval = contribution[timestamp].dt.floor('S') + pd.Timedelta(minutes=int(window_size_mins / 2))
    intervals = pd.IntervalIndex.from_arrays(start_interval, end_interval, closed='left')
    # if closed is change, remember to do the same in every place tha interval is used
    assert not intervals.is_overlapping
    return intervals


def _compute_entropy(x):
    return scipy.stats.entropy(x.value_counts(normalize=True, sort=False), base=10)


def notificationevent(groups):
    return groups.status.value_counts().unstack()


def applicationevent(groups):
    def _get_app_category(apps, category_db) -> pd.Series:
        categories = set()
        apps = apps.unique()
        for a in apps[apps != None]:
            try:  # added
                categories.add(category_db.loc[a, 'category'])  # change: genre -> category
            except KeyError as e:  # added
                # logger.warning(e)
                categories.add('other')  # added

        d = {k: 1 for k in categories}
        d['category_nunique'] = len(categories)
        s = pd.Series(data=d,
                      index=['category_nunique'] + list(category_db['category'].unique()))
        return s

    category_db = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'appcategories.csv'))
    category_db.drop_duplicates(inplace=True)
    category_db.set_index('app_id', verify_integrity=True, inplace=True)
    app_stats = groups['applicationname'].nunique().rename('nunique')
    categories_stat = groups['applicationname'].apply(_get_app_category,
                                                      category_db).unstack()
    entropy = groups['applicationname'].agg(_compute_entropy).rename('entropy_basic')
    out = pd.concat([categories_stat, app_stats, entropy], axis=1, verify_integrity=True).add_prefix('app_')
    out.columns = [c.replace(' ', '_').lower() for c in out.columns]
    out = out.fillna(0)
    return out


def main(path_to_sensor, path_to_answers, output_path):
    logger.info("started")
    start = time.time()
    sensor_name = os.path.basename(path_to_sensor).split('.csv')[0].split('_')[-1]
    logger.info(f"reading sensor data: {sensor_name}")
    sensor = pd.read_csv(path_to_sensor)
    sensor['timestamp'] = pd.to_datetime(sensor['timestamp'], format='%Y%m%d%H%M%S%f')
    logger.info("reading answers data")
    answers = pd.read_csv(path_to_answers, parse_dates=['start_interval', 'end_interval'])

    ### insights start
    logger.info(f'Dataset info:')
    logger.info(f'Number of samples: {len(sensor)}')
    logger.info(f'Number of users: {len(sensor.userid.unique())}')
    logger.info(f'Average sample for each user: {round(len(sensor)/len(sensor.userid.unique()), 2)}')

    sensor['date'] = sensor['timestamp'].dt.date
    user_unique_dates_count = sensor.groupby('userid')['date'].nunique().reset_index(name='unique_dates_count')
    logger.info(f'Average date: {round(user_unique_dates_count["unique_dates_count"].mean(), 2)}')
    sensor = sensor.drop(columns='date')
    ### insights end

    features = []
    for userid in answers.userid.unique():
        logger.warning(f"{userid}/{answers.userid.unique()[-1]}")
        user_answer = answers[answers['userid'] == userid]
        user_sensor = sensor[sensor['userid'] == userid].copy()

        if not len(user_sensor):
            logger.warning(f'Skip user={userid}, no sensor data!')
            continue

        intervals = pd.IntervalIndex.from_arrays(user_answer['start_interval'], user_answer['end_interval'],
                                                 closed='left')
        assert not intervals.is_overlapping

        user_sensor['interval'] = pd.cut(user_sensor.timestamp, intervals, duplicates='raise')

        if user_sensor['interval'].isna().all():
            logger.warning(f'Skip user={userid}, no sensor data is included in any window!')
            continue
        groupbycolumns = ['userid', 'experimentid', 'interval']
        groups = user_sensor.groupby(groupbycolumns, sort=True, group_keys=True, observed=False)

        if sensor_name == 'notificationevent':
            ft = notificationevent(groups)
        elif sensor_name == 'applicationevent':
            ft = applicationevent(groups)
        else:
            raise ("No sensor")

        features.append(ft)
    for ft in features:
        has_dup = ft.index.duplicated().any()
        print(has_dup)

    features = pd.concat(features, axis=0)

    features = features.reset_index(names=['userid', 'experimentid', 'interval'])
    _intervalindex_to_columns(features)
    features.to_csv(os.path.join(output_path, f'{sensor_name}_feature.csv'), index=False)

    logger.info(f"finished in: {round(time.time() - start)} [s]'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s5.log')
    parser.add_argument('-i', '--input', help='path to sensors',
                        # default=os.path.join(PATH_TO_RAW_DATA, 'applicationevent.csv'))
                        default=os.path.join(PATH_TO_RAW_DATA, 'notificationevent.csv'))
    parser.add_argument('-i2', '--input2', help='path to timediary/answers',
                        default=os.path.join(PATH_TO_INTERIM_DATA, 'timediary_feature.csv'))
    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_INTERIM_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.input2, args.output)
