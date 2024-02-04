import pandas as pd
from utils import PATH_TO_INTERIM_DATA, PATH_TO_RAW_DATA, get_logger
import argparse
import os
from sklearn.impute import SimpleImputer


def main(path_to_data, path_to_output_folder):
    df = pd.read_stata(path_to_data)
    # print(df.head())
    columns_to_take = ['userid', 'w1_A01', 'nationality', 'department', 'cohort',
                       'degree', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness',
                       'w1_A09UNITN']
    df = df[columns_to_take]

    logger.info(f'Length of df: {len(df)}')
    df.dropna(inplace=True)
    logger.info(f'Length of df after dropna: {len(df)}')

    if df.isna().any().any():
        logger.info(f"Length of rows with nan values: {len(df)}")

    df.rename(columns={'w1_A01': 'gender', 'w1_A09UNITN': 'location'}, inplace=True)


    ### insights start
    logger.info(f'Raw Dataset info:')
    logger.info(f'Number of samples: {len(df)}')
    logger.info(f'Number of users: {len(df.userid.unique())}')
    logger.info(f'Average sample for each user: {round(len(df) / len(df.userid.unique()), 2)}')
    ### insights end

    df.to_csv(os.path.join(path_to_output_folder, f'demographic.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s2.log')
    parser.add_argument('-i', '--input', help='path to input file',
                        default=os.path.join(PATH_TO_RAW_DATA, 'data4diarynew_ITA.dta'))
    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_INTERIM_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.output)
