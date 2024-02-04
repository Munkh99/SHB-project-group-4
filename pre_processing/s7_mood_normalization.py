import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pre_processing.utils import PATH_TO_PROCESSED_DATA, get_logger


def plot_class_distribution(df, type, output_dir):
    value_counts = df['mood'].value_counts()
    value_counts = value_counts.sort_index()
    plt.figure(figsize=(7, 4))
    COLOR_PASTEL = sns.color_palette()

    plt.bar(value_counts.index, value_counts.values, color=COLOR_PASTEL)
    plt.xlabel('Mood')
    plt.ylabel('Count')
    plt.title('Mood distribution')
    plt.xticks(value_counts.index)
    plt.savefig(os.path.join(output_dir, f'distribution_mood_{type}.png'))
    logger.info(value_counts)
    logger.info(f'viz - distribution_mood_{type}.png')


def main(path_to_data, output_path):
    logger.info("started")
    start = time.time()
    df = pd.read_csv(
        path_to_data,
    )
    ### plot original class distribution
    plot_class_distribution(df, 'original', '/Users/munkhdelger/Documents/unitn/SHB/plots')

    ###

    # Calculate the global mean of the z-scores
    # df_z_score = df.groupby('userid')['mood'].transform(lambda x: (x - x.mean()) / x.std())
    # global_mean_z_score = df_z_score.mean()

    # Function to classify based on domain knowledge and global mean z-score threshold
    # find z score then use global threshold to classify moods . z score normalizer is not robust for homogenous values
    # cuz std is 0 in numinator. so used domain knowledge to classify it when , eg, all values are 5
    def classify_based_on_domain_knowledge_and_z_score(x):

        if x.std() == 0:
            # If standard deviation is zero, use domain knowledge
            if x.unique()[0] in [1, 2]:
                return 'Negative'
            elif x.unique()[0] in [3, 4, 5]:
                return 'Positive'

            # elif x.unique()[0] in [3]:
            #     return 'Neutral'
            # elif x.unique()[0] in [4, 5]:
            #     return 'Positive'
        else:
            # Use z-score normalization and global mean z-score threshold for heterogeneous values
            z_score = (x - x.mean()) / x.std()
            return np.where(z_score <= 0, 'Negative', 'Positive')
            # Set your custom thresholds for classification

            # negative_threshold = -0.3  # Replace with your chosen negative threshold
            # positive_threshold = 0.3  # Replace with your chosen positive threshold
            #
            # # Classify based on thresholds using apply to handle the Series
            # return np.where(z_score < negative_threshold, 'Negative',
            #                 np.where((negative_threshold <= z_score) & (z_score <= positive_threshold), 'Neutral',
            #                          'Positive'))

    df_normalized = df.copy()


    # Apply classification based on domain knowledge or z-score normalization
    df_normalized['mood'] = df_normalized.groupby('userid')['mood'].transform(classify_based_on_domain_knowledge_and_z_score)
    # Display the result
    print(df_normalized.mood.value_counts())
    ### plot original class distribution
    plot_class_distribution(df_normalized, 'normalized', '/Users/munkhdelger/Documents/unitn/SHB/plots')
    ###

    df_normalized.to_csv(os.path.join(output_path, f'final_data_encoded_normalized.csv'), index=False)

    logger.info(f"finished in: {round(time.time() - start)} [s]'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_s6.log')
    parser.add_argument('-i', '--input', help='path to data',
                        default=os.path.join(PATH_TO_PROCESSED_DATA, 'final_data_encoded.csv'))
    parser.add_argument('-o', '--output', help='path to output folder', default=PATH_TO_PROCESSED_DATA)
    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.output)
