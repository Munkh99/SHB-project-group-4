import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from pre_processing.utils import PATH_TO_PROCESSED_DATA, get_logger, PATH_TO_INTERIM_DATA

COLOR_PASTEL = sns.color_palette()

mood_convertion = {
    1: "Very negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Very positive",
}


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


def get_correlation_plot(data, output_folder):
    # df.set_index(['userid', 'answer_time'], inplace=True)
    df = data.drop(columns=['userid', 'experimentid', 'start_interval', 'end_interval'])

    #
    target_variable = 'mood'
    chi_square_results = []
    # Iterate through each column (excluding the target variable)
    for column in df.columns:
        if column != target_variable:
            # Create a contingency table
            contingency_table = pd.crosstab(df[column], df[target_variable])

            # Perform the chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Store the results in a dictionary
            result = {
                'Variable': column,
                'Chi-square statistic': chi2_stat,
                'P-value': p_value,
                'Degrees of freedom': dof
            }

            chi_square_results.append(result)

    # Convert the results to a DataFrame for easier analysis
    results_df = pd.DataFrame(chi_square_results)
    results_df = results_df.sort_values(['P-value'])
    # ss = results_df[results_df['P-value'] <= 0.05]

    temp = ""
    for i, j in zip(results_df['Variable'], results_df['P-value']):
        temp = f'{temp}"\n  {i}:\t{round(j, 4)}'

    logger.info(temp)


def plot_hour_vs_mood_plot(df, output_folder=None, suffix=None):

    custom_colors = {
        5: '#FF5733',  # Warm Red
        4: '#FFA72B',  # Warm Orange
        3: '#FFD035',  # Neutral Yellow
        2: '#229594',  # Cool Green
        1: '#33B5E5'  # Cool Blue
    }

    plt.figure(figsize=(7, 5))
    for mood in df['mood'].unique():
        df_mood = df[df['mood'] == mood]
        hourly_sum_mood = df_mood.groupby('hour')['mood'].count()
        # Calculate percentages
        total_records = hourly_sum_mood.sum()
        hourly_percent_mood = (hourly_sum_mood / total_records) * 100
        plt.plot(hourly_percent_mood.index, hourly_percent_mood.values, label=f'{mood_convertion[mood]}',
                 color=custom_colors[mood])
    plt.xlabel('Hour of the Day')
    plt.ylabel('Percentage of record')
    plt.title(f'Distribution of self report - Mood vs. Hour of the Day ({suffix})')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(output_folder, f'mood_distribution_hour_{suffix}.png'))
    logger.info(f'mood_distribution_hour_{suffix}.png')
    # plt.show()


def plot_gender(df, output_folder=None):
    df_users = df.groupby('userid').first().reset_index()
    female_count = df_users['gender_Female'].sum()
    male_count = df_users['gender_Male'].sum()

    plt.figure(figsize=(5, 4))
    plt.pie([female_count, male_count], autopct='%1.1f%%', startangle=90, colors=COLOR_PASTEL, labels=None)
    plt.legend(loc='center left', labels=['Female', 'Male'], bbox_to_anchor=(1, 0.5), frameon=False)
    plt.title('Gender of students')
    plt.subplots_adjust(left=-0.1)
    logger.info('distribution_gender.png')
    plt.savefig(os.path.join(output_folder, 'distribution_gender.png'))
    # plt.show()

    ###
    degree_columns = df.filter(like='gender_')
    degree_columns[['mood', 'userid']] = df[['mood', 'userid']].copy()

    df_female = degree_columns[degree_columns['gender_Female'] == 1]
    df_male = degree_columns[degree_columns['gender_Male'] == 1]

    result_female = df_female.groupby('userid')['mood'].value_counts(normalize=True).unstack(fill_value=0)
    result_male = df_male.groupby('userid')['mood'].value_counts(normalize=True).unstack(fill_value=0)

    proportion_female = result_female.mean()
    proportion_male = result_male.mean()

    bar_width = 0.35
    index = np.arange(len(proportion_female))
    plt.figure(figsize=(7, 5))
    plt.bar(index, proportion_female, width=bar_width, label='Female')
    plt.bar(index + bar_width, proportion_male, width=bar_width, label='Male')

    # Adding labels and title
    plt.xlabel('Mood')
    plt.ylabel('Proportion')
    plt.title('Proportion of Mood in gender')
    # plt.xticks(index + bar_width / 2, ["Negative", "Positive"])
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'distribution_gender_by_mood.png'))
    logger.info('distribution_gender_by_mood.png')
    # plt.show()



def plot_age(df, output_folder=None):
    df_users = df.groupby('userid').first().reset_index()
    cohort_columns = df_users.filter(like='cohort')
    cohort_counts = cohort_columns.sum()

    labels = [f'{round(c / cohort_counts.sum(), 2) * 100}%    {str(l).replace("cohort_", "")}' for c, l in
              zip(cohort_counts, cohort_counts.index)]
    plt.figure(figsize=(5, 4))
    plt.pie(cohort_counts, autopct='%1.1f%%', startangle=90, colors=COLOR_PASTEL)
    plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.title("Age of students")
    plt.subplots_adjust(left=-0.2)
    plt.savefig(os.path.join(output_folder, 'distribution_age.png'))
    # plt.show()


def plot_degree(df, output_folder=None):
    # vizualizein demogrpahic values so need to groupby users
    df_users = df.groupby('userid').first().reset_index()
    degree_columns = df_users.filter(like='degree')
    degree_counts = degree_columns.sum()

    labels = [f'{str(l).replace("degree_", "")}' for c, l in
              zip(degree_counts, degree_counts.index)]
    plt.figure(figsize=(5, 4))
    plt.pie(degree_counts, autopct='%1.1f%%', startangle=90, colors=COLOR_PASTEL)
    plt.legend(labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.title("Education of Students")
    plt.subplots_adjust(left=-0.1)
    plt.savefig(os.path.join(output_folder, 'distribution_degree.png'))
    logger.info('distribution_degree.png')
    # plt.show()

    # vizualizein timediary so we need whole data

    degree_columns = df.filter(like='degree')
    degree_columns[['mood', 'userid']] = df[['mood', 'userid']].copy()

    df_bs = degree_columns[degree_columns['degree_BSc'] == 1]
    df_ms = degree_columns[degree_columns['degree_MSc'] == 1]

    result_bs = df_bs.groupby('userid')['mood'].value_counts(normalize=True).unstack(fill_value=0)
    result_ms = df_ms.groupby('userid')['mood'].value_counts(normalize=True).unstack(fill_value=0)

    proportion_bs = result_bs.mean()
    proportion_ms = result_ms.mean()

    bar_width = 0.35
    index = np.arange(len(proportion_bs))
    plt.figure(figsize=(7, 5))
    plt.bar(index, proportion_bs, width=bar_width, label='BSc')
    plt.bar(index + bar_width, proportion_ms, width=bar_width, label='MSc')

    # Adding labels and title
    plt.xlabel('Mood')
    plt.ylabel('Proportion')
    plt.title('Proportion of Mood in BSc and MSc')
    # plt.xticks(index + bar_width / 2, ['Negative', 'Positive'])
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'distribution_degree_by_mood.png'))
    # plt.show()
    plt.legend(loc='upper right')

    return


def analyze_places(data, output_folder=None):
    df = data.filter(regex='where_')
    df['mood'] = data['mood'].copy()
    # udf = df.groupby(['mood']).agg(sum)
    # timediary_percent = udf.div(udf.sum(axis=1), axis=0) * 100

    custom_colors = {
        5: '#FF5733',  # Warm Red
        4: '#FFA72B',  # Warm Orange
        3: '#FFD035',  # Neutral Yellow
        2: '#51FFA8',  # Cool Green
        1: '#33B5E5'  # Cool Blue
    }
    mood_dict = {}
    unique_moods = df['mood'].unique()
    # Calculate the number of rows and columns
    num_rows = (len(unique_moods) + 1) // 2  # Add 1 to round up in case of an odd number of moods
    num_cols = 2

    # Create subplots in a 2x3 grid
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(14, 6))

    # Flatten the axes array to iterate over it
    axes = axes.flatten()

    # Iterate through each mood
    for i, mood in enumerate(unique_moods):
        udf = df[df['mood'] == mood]
        udf = udf.groupby(['mood']).agg(sum)
        timediary_percent = udf.div(udf.sum(axis=1), axis=0) * 100
        sorted_timediary_percent = timediary_percent.sum(axis=0).sort_values(ascending=False)

        # Get the top 5 columns
        top_5_columns = sorted_timediary_percent.head(5)
        top_5_columns = top_5_columns[::-1]

        # Plot on the respective subplot as a horizontal bar plot with custom colors
        ax = axes[i]
        bars = top_5_columns.plot(kind='barh', color=custom_colors[mood], ax=ax)
        ax.set_title(f'Top 5 Places : {mood_convertion[mood]}')
        ax.set_xlabel('Percentage')
        ax.set_xlim(0, 85)

        # Add percentage values on the bars
        for bar in bars.patches:
            width = bar.get_width()
            label_x_pos = width + 0.2  # Adjust this value for the correct positioning
            label_y_pos = bar.get_y() + bar.get_height() / 2
            ax.text(label_x_pos, label_y_pos, f'{width:.2f}%', va='center')

        mood_dict[mood] = top_5_columns

    # fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'distribution_place_by_mood.png'))
    # Show the plots
    # plt.show()
    return None


def plot_mood_vs_app_category(data, output_folder):
    df = data.filter(regex='app_')
    df = df.drop(columns=['app_category_nunique', 'app_entropy_basic', 'app_nunique',
                          'app_other'])
    df['mood'] = data['mood']

    app_column_names = df.columns.tolist()
    app_column_names.remove('mood')
    print(app_column_names)

    proportions_df = pd.DataFrame()
    for mood in reversed(sorted(df['mood'].unique())):
        df_mood = df[df['mood'] == mood]
        column_sum = df_mood.drop(['mood'], axis=1).sum()
        proportions = column_sum.div(column_sum.sum()) * 100

        # Add the proportions to the DataFrame with mood as the index
        proportions_df[mood] = proportions
        # proportions_df[mood] = column_sum

    # Transpose the DataFrame to have moods as rows and apps as columns
    proportions_df = proportions_df.T

    x_axis_labels = ['internet_&_social..', 'productivity_&_a..',
                     'shopping', 'media', 'maps_&_travel', 'health', 'games', 'food_&_drink']

    # Create a heatmap yticklabels=['Positive', 'Negative']
    plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(proportions_df,
                          xticklabels=x_axis_labels,
                          yticklabels=['Very positive','Positive', 'Neutral', 'Negative', 'Very negative'],
                          annot=True,
                          cmap="Blues", fmt=".1f",
                          cbar_kws={'label': 'Proportion (%)'})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=30, ha='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, ha='right')
    plt.title('Proportion of app categories in mood')
    plt.subplots_adjust(left=0.2, bottom = 0.2)
    plt.savefig(os.path.join(output_folder, 'heatmap_mood_vs_app_cat.png'))
    logger.info('heatmap_mood_vs_app_cat.png')
    # plt.show()
    return


def plot_mood_vs_notif(data, output_folder, suffix=None):
    df = data.filter(like='notification_')
    correlation = df['notification_posted'].corr(data['mood'])
    correlation2 = df['notification_removed'].corr(data['mood'])

    print(f"Pearson Correlation Coefficient: {correlation}")
    print(f"Pearson Correlation Coefficient: {correlation2}")

    df = data.filter(like='notification_')
    df['mood'] = data['mood']

    cnt = df.groupby('mood').sum()
    # Calculate proportions for each row
    proportions = cnt.div(cnt.sum(axis=1), axis=0)

    proportions_posted = proportions['notification_posted']
    proportions_removed = proportions['notification_removed']


    bar_width = 0.35

    index = np.arange(len(proportions_posted))
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.bar(index, proportions_posted, width=bar_width, label='Notification_posted')
    plt.bar(index + bar_width, proportions_removed, width=bar_width, label='Notification_removed')

    plt.xlabel('Notification')
    plt.ylabel('Mood')
    plt.title('Proportion of Notification status in Mood')
    # plt.xticks(index + bar_width / 2, proportions.columns)
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'notif_x_mood.png'))

    ### Notif x time

    df = data.filter(like='notification_')
    df['hour'] = data['hour']

    custom_colors = {
        5: '#FF5733',  # Warm Red
        4: '#FFA72B',  # Warm Orange
        3: '#FFD035',  # Neutral Yellow
        2: '#229594',  # Cool Green
        1: '#33B5E5'  # Cool Blue
    }

    plt.figure(figsize=(7, 5))

    hourly_notif_posted = df.groupby('hour')['notification_posted'].sum()
    hourly_notif_removed = df.groupby('hour')['notification_removed'].sum()

    total_records_posted = hourly_notif_posted.sum()
    total_records_removed = hourly_notif_removed.sum()

    hourly_percent_posted = (hourly_notif_posted / total_records_posted) * 100
    hourly_percent_removed = (hourly_notif_removed / total_records_removed) * 100

    plt.plot(hourly_percent_posted.index, hourly_percent_posted.values, label=f'Notification posted',
             color='#FFA72B')
    plt.plot(hourly_percent_removed.index, hourly_percent_removed.values, label=f'Notification removed',
             color='#229594')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Percentage of record')
    plt.title(f'Distribution: Notification vs. Hour of the Day ({suffix}) ')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_folder, f'notification_distribution_hour_{suffix}.png'))
    logger.info(f'notification_distribution_hour_{suffix}.png')

    return


def main(path_to_data, path_to_data_encoded, path_to_output):
    data = pd.read_csv(
        os.path.join(path_to_data)
    )

    data_encoded = pd.read_csv(
        os.path.join(path_to_data_encoded)
    )

    workdays_df = data_encoded[
        (data_encoded['week_Monday'] == 1) | (data_encoded['week_Tuesday'] == 1) | (
                    data_encoded['week_Wednesday'] == 1) |
        (data_encoded['week_Thursday'] == 1) | (data_encoded['week_Friday'] == 1)]
    weekdays_df = data[
        (data_encoded['week_Saturday'] == 1) | (data_encoded['week_Sunday'] == 1)]
    assert (len(data_encoded) == len(workdays_df) + len(weekdays_df))

    get_correlation_plot(data, path_to_output)
    #
    plot_gender(data_encoded, path_to_output)
    plot_age(data_encoded, path_to_output)
    plot_degree(data_encoded, path_to_output)
    #
    # # all, workdays, weekdays
    #
    # plot_hour_vs_mood_plot(data_encoded, path_to_output, suffix='overall')
    # plot_hour_vs_mood_plot(workdays_df, path_to_output, suffix='workdays')
    # plot_hour_vs_mood_plot(weekdays_df, path_to_output, suffix='weekdays')

    # plot_mood_vs_app_category(data, path_to_output)  # heatmap + barplots
    plot_mood_vs_notif(data, path_to_output, suffix='all')
    # plot_mood_vs_notif(workdays_df, path_to_output, suffix='workdays')
    # plot_mood_vs_notif(weekdays_df, path_to_output, suffix='weekdays')

    # analyze_places(data, path_to_output)
    # viz.
    print("finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_viz.log')
    parser.add_argument('-o', '--output', help='path to plot directory', default='plots')
    parser.add_argument('-i', '--input', help='path to input directory',
                        default=os.path.join(PATH_TO_PROCESSED_DATA, 'final_data.csv'))
    parser.add_argument('-ii', '--input_encoded', help='path to input directory2',
                        default=os.path.join(PATH_TO_PROCESSED_DATA, 'final_data_encoded.csv'))

    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input, args.input_encoded, args.output)
