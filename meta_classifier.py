import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import torch
from collections import Counter

data = torch.load("now-merge.pt")
all_features = []
dataset_names = list(data.keys())

for dataset_name, features in data.items():
    all_features.append(features.numpy())

features = np.vstack(all_features)
num_datasets = len(dataset_names)
features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

valid_labels = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
label_remap_dict = {orig: new for new, orig in enumerate(valid_labels)}

y_raw = features[:, -1:].astype(int)
y = np.vectorize(label_remap_dict.get)(y_raw)
y = y.reshape(-1, 1)
X = features[:, :-1]

raw_mapping = torch.load("dataset_algorithm_mapping.pt")
dataset_algorithm_mapping = {}

for dataset, acc_map in raw_mapping.items():
    new_map = {}
    for orig_label, acc in acc_map.items():
        if orig_label in label_remap_dict:
            new_label = label_remap_dict[orig_label]
            new_map[new_label] = acc
    dataset_algorithm_mapping[dataset] = new_map

results = []

for dataset_idx, dataset_name in enumerate(dataset_names):
    print(f"\n=== {dataset_idx + 1}/{num_datasets} ===")

    start_idx = dataset_idx * 20
    end_idx = start_idx + 20

    X_train_raw = np.vstack((X[:start_idx], X[end_idx:]))
    y_train_raw = np.vstack((y[:start_idx], y[end_idx:]))
    X_test = X[start_idx:end_idx]
    y_test = y[start_idx:end_idx]

    X_train = X_train_raw
    y_train = y_train_raw.flatten()

    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        random_state=None
    )
    rf_clf.fit(X_train, y_train)

    y_test_pred_prob = rf_clf.predict_proba(X_test)
    top_k_preds = np.argsort(y_test_pred_prob, axis=1)[:, -3:]

    accuracy_mapping = dataset_algorithm_mapping.get(dataset_name, {})


    top3_labels_accuracy = sorted(accuracy_mapping.items(), key=lambda x: x[1], reverse=True)[:3]
    true_top3_accuracies = [acc for _, acc in top3_labels_accuracy]

    rmv_top1 = []
    rmv_avg = []
    rmv_max = []
    top1_pred_labels = []

    if accuracy_mapping:
        for i in range(20):
            true_label = y_test[i, 0]
            pred_top1 = top_k_preds[i, -1]
            pred_top2 = top_k_preds[i, -2]
            pred_top3 = top_k_preds[i, -3]

            pred_top1_accuracy = accuracy_mapping.get(pred_top1, 0)
            pred_top2_accuracy = accuracy_mapping.get(pred_top2, 0)
            pred_top3_accuracy = accuracy_mapping.get(pred_top3, 0)

            top1_pred_labels.append(pred_top1)

            rmv_top1_value = pred_top1_accuracy / true_top3_accuracies[0] if pred_top1_accuracy > 0 else 0
            rmv_top1.append(rmv_top1_value)

            avg_rmv_values = []
            for j, true_acc in enumerate(true_top3_accuracies):
                pred_acc = [pred_top1_accuracy, pred_top2_accuracy, pred_top3_accuracy][j]
                if pred_acc > 0:
                    avg_rmv_values.append(pred_acc / true_acc)
            if avg_rmv_values:
                rmv_avg.append(np.mean(avg_rmv_values))

            max_pred_accuracy = max(pred_top1_accuracy, pred_top2_accuracy, pred_top3_accuracy)
            rmv_max_value = max_pred_accuracy / true_top3_accuracies[0] if max_pred_accuracy > 0 else 0
            rmv_max.append(rmv_max_value)

    if top1_pred_labels:
        majority_vote_label = Counter(top1_pred_labels).most_common(1)[0][0]
        majority_vote_accuracy = accuracy_mapping.get(majority_vote_label, 0)
        majority_top1_rmv = majority_vote_accuracy / true_top3_accuracies[0] if true_top3_accuracies[0] > 0 else 0
    else:
        majority_vote_label = -1
        majority_top1_rmv = 0

    weighted_counter = Counter()
    for i in range(20):
        pred_top3 = top_k_preds[i]
        weighted_counter[pred_top3[-1]] += 3
        weighted_counter[pred_top3[-2]] += 2
        weighted_counter[pred_top3[-3]] += 1
    weighted_vote_label = weighted_counter.most_common(1)[0][0]
    weighted_vote_accuracy = accuracy_mapping.get(weighted_vote_label, 0)
    weighted_top1_rmv = weighted_vote_accuracy / true_top3_accuracies[0] if true_top3_accuracies[0] > 0 else 0

    avg_proba = np.mean(y_test_pred_prob, axis=0)
    proba_vote_label = np.argmax(avg_proba)
    proba_vote_accuracy = accuracy_mapping.get(proba_vote_label, 0)
    proba_top1_rmv = proba_vote_accuracy / true_top3_accuracies[0] if true_top3_accuracies[0] > 0 else 0

   
    top1_rmv = np.mean(rmv_top1) if rmv_top1 else 0
    avg_rmv = np.mean(rmv_avg) if rmv_avg else 0
    max_rmv = np.mean(rmv_max) if rmv_max else 0

    results.append([
        dataset_name, top1_rmv, avg_rmv, max_rmv,
        majority_vote_label, majority_top1_rmv,
        weighted_vote_label, weighted_top1_rmv,
        proba_vote_label, proba_top1_rmv
    ])

df_results = pd.DataFrame(results, columns=[
    "Dataset_Name", "Top1_RMV", "Avg_RMV", "Max_RMV",
    "MajorityTop1_Label", "MajorityTop1_RMV",
    "WeightedTop3_Label", "WeightedTop3_RMV",
    "ProbaAvg_Label", "ProbaAvg_RMV"
])
df_results.to_csv("rmv_results.csv", index=False)
joblib.dump(rf_clf, "best_rf_model.pkl")

