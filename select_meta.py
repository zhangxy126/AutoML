import torch
import pandas as pd
import numpy as np

meta_file = "meta-features/cleaned_file.xlsx"
df = pd.read_excel(meta_file)

selected_features = np.load("DQN/results/selected_features.npy")

dataset_meta_features = {}

for index, row in df.iterrows():
    dataset_name = row['Dataset']
    meta_features = row[1:].values 
    selected_meta_features = meta_features[selected_features]  
    
    selected_meta_features = selected_meta_features.astype(np.float32)
    
    dataset_meta_features[dataset_name] = torch.tensor(selected_meta_features, dtype=torch.float32)

torch.save(dataset_meta_features, "selected_meta_features.pt")

print("Filtered meta features have been saved to 'selected_meta_features.pt'.")
