import argparse
import os
from io import StringIO

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from pre_processing.utils import PATH_TO_PROCESSED_DATA, get_logger

import argparse
import os
from io import StringIO
from sklearn.metrics import f1_score
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from pre_processing.utils import PATH_TO_PROCESSED_DATA, get_logger


class classifier:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42, n_estimators=100)

        self.results = None

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # classification_rep = classification_report(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        self.results = {
            "accuracy": accuracy,
            "f1": f1,
            "confusion_table": cm.tolist()
        }

    def get_results(self):
        return self.results

    def save_results(self, filename):
        if self.results is not None:
            with open(filename, 'w') as file:
                file.write(f"Random Forest Results:\n")
                file.write(f"Accuracy: {self.results['accuracy']}\n")
                file.write(f"Classification Report:\n{self.results['classification_report']}")


class Dataloader:
    def __init__(self, path_to_data):
        self.df = pd.read_csv(path_to_data)

    def get_userids(self):
        return self.df.userid.unique()

    def get_train_test(self, userid):
        udf = self.df[self.df['userid'] == userid]
        features = udf.drop(columns=['mood', 'userid', 'experimentid', 'start_interval', 'end_interval'])
        target = udf['mood']
        if len(target.value_counts()) == 1 or target.value_counts()[1] == 1:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,
                                                                random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2,
                                                                random_state=42)
        return X_train, X_test, y_train, y_test


def main(path_to_data):
    dataloader = Dataloader(path_to_data=path_to_data)

    global_result = []
    for userid in dataloader.get_userids():
        X_train, X_test, y_train, y_test = dataloader.get_train_test(userid)
        model = classifier()
        model.train(X_train, y_train)
        model.evaluate(X_test, y_test)
        results = model.get_results()
        global_result.append(results)
        logger.info(f"Userid: {userid}, Accuracy: {results['accuracy']}, # of feature: {len(X_train.columns)}")

    ### get aggregated results:
    df = pd.DataFrame(global_result)

    df['confusion_table'] = df['confusion_table'].apply(np.array)

    def transform_scalar_to_2x2(matrix):
        if matrix.size == 1:
            return np.array([[0, 0], [0, matrix[0, 0]]])
        else:
            return matrix

    df['confusion_table'] = df['confusion_table'].apply(transform_scalar_to_2x2)
    # Sum of all values in each position of 'Confusion_table'
    sum_confusion_table = np.sum(df['confusion_table'], axis=0)

    logger.info("Sum of Confusion_table:")
    logger.info(sum_confusion_table.tolist())
    logger.info(f'Average accuracy {df["accuracy"].mean()}')
    logger.info(f'Average f1 {df["f1"].mean()}')

    logger.info("finished")
    # feature_importance = model.get_ft_importance()
    # feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
    # feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    # threshold = feature_importance_df['Importance'].mean()
    # # threshold = 0.02
    # logger.info(f"Threshold: {threshold}")
    # selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]
    # import seaborn as sns
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='Importance', y='Feature', data=selected_features)
    # plt.title('Feature Importance')
    # plt.show()
    #
    # X_train_selected = X_train[selected_features['Feature']]
    # X_test_selected = X_test[selected_features['Feature']]
    #
    # model = classifier()
    # model.train(X_train_selected, y_train)
    # model.evaluate(X_test_selected, y_test)

    # logger.info(f"Accuracy: {results['accuracy']}, # of feature: {len(X_train_selected.columns)}")


if __name__ == '__main__':
    import warnings

    # Mute FutureWarnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_main_per_user.log')
    parser.add_argument('-i', '--input', help='path to input file',
                        default=os.path.join(PATH_TO_PROCESSED_DATA, 'final_data_encoded_normalized.csv'))

    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input)
    # Re-enable warnings (optional, but recommended)
    warnings.filterwarnings("default", category=FutureWarning)
