import torch
import pandas as pd

# read .pt 文件
meta_features_file = "selected_meta_features.pt"
image_features_file = "pic/256image_features.pt"
top3_labels_file = "top3_labels.pt"

meta_features = torch.load(meta_features_file)  
image_features = torch.load(image_features_file)  
top3_labels = torch.load(top3_labels_file)

combined_features = {}

for dataset_name in meta_features.keys():
    try:
        meta_feature = meta_features[dataset_name]
        image_feature = image_features[dataset_name]
        top3_label = top3_labels[dataset_name]
        top1_label = top3_label[0].unsqueeze(0)
        if image_feature.shape[0] != 20 or image_feature.shape[1] != 256:
            print(f"Warning: {dataset_name} has incorrect image feature shape.")
            continue

        combined_data_list = []
        for i in range(image_feature.shape[0]):
            combined_data = torch.cat([meta_feature, image_feature[i], top1_label])
            combined_data_list.append(combined_data)

        combined_features[dataset_name] = torch.stack(combined_data_list)
    
    except KeyError as e:
        print(f"Error: {dataset_name} not found in one of the files. Skipping this dataset.")
        continue

torch.save(combined_features, "now-merge.pt")

print("Combined features have been saved to 'now-merge.pt'.")
