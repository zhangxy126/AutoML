import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import joblib

class RewardCalculator:
    def __init__(self):
        self.base_model = None
        self.base_accuracy = None

    def train_base_model(self, X, y):
        self.base_accuracy = self.cross_val_top3_accuracy(X, y)
        self.base_model = RandomForestClassifier(n_estimators=100)
        self.base_model.fit(X, y)

    def cross_val_top3_accuracy(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            y_pred_top3 = np.argsort(y_pred_proba, axis=1)[:, -3:]

            acc = self.compute_top3_accuracy(y_test, y_pred_top3)
            accuracies.append(acc)

        return np.mean(accuracies)

    def compute_top3_accuracy(self, y_true, y_pred_top3):
        correct_predictions = 0
        total_samples = len(y_true) // 3

        for i in range(total_samples):
            true_top3 = {y_true[i * 3], y_true[i * 3 + 1], y_true[i * 3 + 2]}
            pred_top3 = set(y_pred_top3[i * 3])

            if true_top3 & pred_top3:
                correct_predictions += 1

        return correct_predictions / total_samples

    def compute_modified_accuracy(self, X, y, removed_feature_index):
        X_modified = np.delete(X, removed_feature_index, axis=1)
        return self.cross_val_top3_accuracy(X_modified, y)
    def get_rewards(self, X, y):
        rewards = np.zeros(X.shape[1], dtype=np.float32)

        for i in range(X.shape[1]):
            modified_accuracy = self.compute_modified_accuracy(X, y, removed_feature_index=i)
            rewards[i] = self.base_accuracy - modified_accuracy
         
        return rewards


meta_features_file = '../../meta-features/cleaned_file.xlsx'
meta_df = pd.read_excel(meta_features_file)

labels_file = '../../dataset/top3_algorithms_avg.xlsx'
labels_df = pd.read_excel(labels_file)

meta_df.rename(columns={'Dataset': 'Dataset_Name'}, inplace=True)
labels_df.rename(columns={'Dataset': 'Dataset_Name'}, inplace=True)

expanded_data = []
for _, row in labels_df.iterrows():
    dataset_name = row['Dataset_Name']
    for i in range(1, 4):
        expanded_data.append([dataset_name, row[f'Top{i}_Category']])

expanded_labels_df = pd.DataFrame(expanded_data, columns=['Dataset_Name', 'Label'])

merged_df = pd.merge(meta_df, expanded_labels_df, on='Dataset_Name', how='inner')

X_expanded = merged_df.iloc[:, 1:-1].values.astype(np.float32)
y_expanded = merged_df.iloc[:, -1].values.astype(np.int32)

X_expanded[np.isinf(X_expanded)] = 1e5
X_expanded[X_expanded == -np.inf] = -1e5
X_expanded = np.nan_to_num(X_expanded, nan=0.0)

reward_calculator = RewardCalculator()
reward_calculator.train_base_model(X_expanded, y_expanded)
rewards = reward_calculator.get_rewards(X_expanded, y_expanded)

joblib.dump(reward_calculator, 'reward_model.pkl')
np.save('rewards.npy', rewards)
