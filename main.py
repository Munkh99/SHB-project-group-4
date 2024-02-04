import argparse
import os
import numpy as np
from io import StringIO
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from pre_processing.utils import PATH_TO_PROCESSED_DATA, get_logger


class classifier:
    def __init__(self, model):
        self.model = model
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

    def get_ft_importance(self):
        return self.model.feature_importances_

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

    def get_train_test(self):
        # features = self.df.drop(columns=['mood', 'userid', 'experimentid'])
        features = self.df.drop(columns=['mood', 'userid', 'experimentid', 'start_interval', 'end_interval'])
        target = self.df['mood']
        X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

def get_confusion_table(results):
    logger.info("Confusion_table:")
    logger.info(results['confusion_table'])
    import seaborn as sns
    sns.set(font_scale=1.2)
    sns.heatmap(results['confusion_table'], annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join('/Users/munkhdelger/Documents/unitn/SHB/plots', 'ml_confusion_matrix_all.png'))
    # plt.show()




def main(path_to_data):
    dataloader = Dataloader(path_to_data=path_to_data)

    X_train, X_test, y_train, y_test = dataloader.get_train_test()
    m = RandomForestClassifier(random_state=42, n_estimators=100)
    model = classifier(m)
    model.train(X_train, y_train)
    model.evaluate(X_test, y_test)
    results = model.get_results()
    logger.info(f'Number of features :{len(X_train.columns)}')
    logger.info(f'Average accuracy {results["accuracy"].mean()}')
    logger.info(f'Average f1 {results["f1"].mean()}')

    get_confusion_table(results)

    feature_importance = model.get_ft_importance()
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    threshold = feature_importance_df['Importance'].mean()
    print(threshold)

    # plt.bar(range(len(feature_importance)), sorted(feature_importance, reverse=True), align='center')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Feature Importance')
    # plt.yticks(np.arange(0, 0.05, 0.005))
    # plt.show()

    threshold = 0.01
    logger.info(f"Threshold: {threshold}")
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]
    print(selected_features)
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    sns.set(rc={'figure.figsize': (6, 8), 'axes.labelsize': 14})
    plt.figure(figsize=(6, 8))
    sns.barplot(x='Importance', y='Feature', data=selected_features)
    plt.title('Feature Importance')
    plt.subplots_adjust(left=0.3)  # You can adjust the value to control the left margin
    plt.savefig(os.path.join('/Users/munkhdelger/Documents/unitn/SHB/plots', 'ml_feature_importance_all.png'))

    X_train_selected = X_train[selected_features['Feature']]
    X_test_selected = X_test[selected_features['Feature']]

    m = RandomForestClassifier(random_state=42, n_estimators=100)
    model = classifier(m)
    model.train(X_train_selected, y_train)
    model.evaluate(X_test_selected, y_test)

    results = model.get_results()
    logger.info("Confusion_table:")
    logger.info(results['confusion_table'])
    logger.info(f'Number of features :{len(X_train_selected.columns)}')
    logger.info(f'Average accuracy {results["accuracy"].mean()}')
    logger.info(f'Average f1 {results["f1"].mean()}')



if __name__ == '__main__':
    import warnings

    # Mute FutureWarnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs', help='path to logging file', default='log_main.log')
    parser.add_argument('-i', '--input', help='path to input file',
                        default=os.path.join(PATH_TO_PROCESSED_DATA, 'final_data_encoded_normalized.csv'))

    args = parser.parse_args()
    logger = get_logger(os.path.basename(__file__), args.logs)
    main(args.input)
    # Re-enable warnings (optional, but recommended)
    warnings.filterwarnings("default", category=FutureWarning)
