import pandas as pd
from utils import PATH_TO_INTERIM_DATA, PATH_TO_RAW_DATA, get_logger
import argparse
import os


def main(path_to_data, path_to_output_folder):
    df = pd.read_stata(path_to_data)
    columns_to_take = ['pilot', 'id', 'date_not', 'first2w', 'week',
                       'what', 'where', 'withw', 'mood', 'sleep', 'expectday',
                       'howwasday', 'uniproblem'
                       ]
    df = df[columns_to_take]

    column_name_mapping = {
        'pilot': 'experimentid',
        'id':'userid',
        'date_not': 'timestamp',
    }
    df.rename(columns=column_name_mapping, inplace=True)

    df['userid'] = df['userid'].astype(int)

    # since we are only considering time points which has answers, we are excluding empty time points
    df_mood = df[(df['mood'] != 'No information') &
                 (df['mood'] != 'Travel') &
                 (df['mood'] != 'Expired') &
                 (df['mood'] != 'Not answer')].copy()

    df_mood['sleep'].ffill(inplace=True)
    df_mood['expectday'].ffill(inplace=True)
    df_mood['howwasday'].ffill(inplace=True)
    df_mood['uniproblem'].ffill(inplace=True)

    # re labeling mood

    replacement = {'Happy': 5.0, 'Sad': 1.0}
    df_mood['mood'] = df_mood['mood'].replace(replacement)
    df_mood["sleep"] = df_mood['sleep'].replace(replacement)
    df_mood["expectday"] = df_mood['expectday'].replace(replacement)
    df_mood["howwasday"] = df_mood['sleep'].replace(replacement)


    # relpacing nan values by "No information"
    df_mood['sleep'] = df_mood['sleep'].fillna('No information')
    df_mood['expectday'] = df_mood['expectday'].fillna('No information')
    df_mood['howwasday'] = df_mood['howwasday'].fillna('No information')
    df_mood['uniproblem'] = df_mood['uniproblem'].fillna('No information')

    logger.info(len(df_mood))
    df_mood = df_mood[(df_mood['sleep'] != 'No information') & (df_mood['sleep'] != 'Expired')]
    df_mood = df_mood[(df_mood['expectday'] != 'No information') & (df_mood['expectday'] != 'Expired')]
    df_mood = df_mood[(df_mood['howwasday'] != 'No information') & (df_mood['howwasday'] != 'Expired')]
    df_mood = df_mood[(df_mood['uniproblem'] != 'No information') & (df_mood['uniproblem'] != 'Expired')]

    logger.info(len(df_mood))
    df_mood.to_csv(os.path.join(path_to_output_folder, f'timediary.csv'), index=False)

    ### insights start
    logger.info(f'Dataset info:')
    logger.info(f'Number of samples: {len(df_mood)}')
    logger.info(f'Number of users: {len(df_mood.userid.unique())}')
    logger.info(f'Average sample for each user: {round(len(df_mood)/len(df_mood.userid.unique()), 2)}')

    df_mood['date'] = df_mood['timestamp'].dt.date
    user_unique_dates_count = df_mood.groupby('userid')['date'].nunique().reset_index(name='unique_dates_count')
    logger.info(f'average dates: {round(user_unique_dates_count["unique_dates_count"].mean(), 2)}')
    ### insights end

    logger.info("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s1.log')
    parser.add_argument('-i', '--input', help='path to input file',
                        default=os.path.join(PATH_TO_RAW_DATA, 'td_ita.dta'))
    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_INTERIM_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.output)
