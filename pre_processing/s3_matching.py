import time
import pandas as pd
from utils import PATH_TO_INTERIM_DATA, PATH_TO_RAW_DATA, get_logger, PATH_TO_PROCESSED_DATA, parse_interval, \
    _intervalindex_to_columns
import argparse
import os


def compute_windows_intervals(contribution: pd.DataFrame, window_size_mins: int) -> pd.IntervalIndex:
    timestamp = 'timestamp'
    start_interval = contribution[timestamp].dt.floor('S') - pd.Timedelta(minutes=int(window_size_mins / 2))
    end_interval = contribution[timestamp].dt.floor('S') + pd.Timedelta(minutes=int(window_size_mins / 2))
    intervals = pd.IntervalIndex.from_arrays(start_interval, end_interval, closed='left')
    assert not intervals.is_overlapping
    return intervals


def main(path_to_data, path_to_timediary, path_to_output_folder):
    logger.info("started")
    start = time.time()
    df_demographic = pd.read_csv(path_to_data)
    dtype_mapping = {
        'expectday': str,
        'sleep': str,
        'uniproblem': str,
        'howwasday': str,

    }
    df_timediary = pd.read_csv(path_to_timediary, dtype=dtype_mapping, parse_dates=['timestamp'])

    df_merged = []
    for userid in df_timediary["userid"].unique():
        udf_time = df_timediary[df_timediary["userid"] == userid].copy()
        udf_demog = df_demographic[df_demographic["userid"] == userid]
        if len(udf_demog) == 0:
            logger.info(f"no demographic data for user {userid}")

        intervals = compute_windows_intervals(udf_time, window_size_mins=30)
        udf_time['interval'] = pd.cut(udf_time.timestamp, intervals, duplicates='raise')
        merged_df = udf_demog.merge(udf_time, on='userid', how='inner')
        df_merged.append(merged_df)

    df_merged = pd.concat(df_merged, axis=0)
    logger.info(f'{len(df_merged.userid.unique())}, {len(df_merged)}')
    df_merged.sort_values(['userid', 'timestamp'], inplace=True)

    logger.info(
        f'{len(df_merged.userid.unique())}, {len(df_merged)}, {round(len(df_merged) / len(df_merged.userid.unique()), 2)}')
    df_merged = df_merged.drop(columns=['timestamp'])
    _intervalindex_to_columns(df_merged)
    df_merged.to_csv(os.path.join(path_to_output_folder, f'timediary_feature.csv'), index=False)

    logger.info(len(df_merged.userid.unique()))
    logger.info(f"finished in: {round(time.time() - start)} [s]'")


if __name__ == '__main__':
    print(PATH_TO_RAW_DATA)

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s3.log')
    parser.add_argument('-i', '--input', help='path to input file',
                        default=os.path.join(PATH_TO_INTERIM_DATA, 'demographic.csv'))
    parser.add_argument('-i2', '--input2', help='path to input file',
                        default=os.path.join(PATH_TO_INTERIM_DATA, 'timediary.csv'))
    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_INTERIM_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.input2, args.output)
