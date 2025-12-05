import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.dat'):
        df = pd.read_csv(file_path, delim_whitespace=True)  
    else:
        raise ValueError(f"error file: {file_path}")

    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1].values  
    
    for col in features.columns:
        if features[col].dtype == 'O': 
            encoder = LabelEncoder()
            features[col] = encoder.fit_transform(features[col])

    if labels.dtype == 'O': 
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

    features = features.fillna(features.mean())

    features[features == np.inf] = np.nan
    features[features == -np.inf] = np.nan

    features = features.fillna(features.mean())

    return features.values, labels

def create_image(X, labels, output_path, img_width=256, img_height=256, random_state=None):
    tsne = TSNE(n_components=2, perplexity=6, learning_rate=200, random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    unique_labels = np.unique(labels)
    label_colors = {label: (np.random.randint(0, 256),
                            np.random.randint(0, 256),
                            np.random.randint(0, 256)) for label in unique_labels}

    image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  

    x_min, x_max = X_embedded[:, 0].min(), X_embedded[:, 0].max()
    y_min, y_max = X_embedded[:, 1].min(), X_embedded[:, 1].max()

    def normalize(value, min_val, max_val, scale):
        return int((value - min_val) / (max_val - min_val) * (scale - 1))

    drawn_positions = set()

    for i, (x, y) in enumerate(X_embedded):
        x_norm = normalize(x, x_min, x_max, img_width)
        y_norm = normalize(y, y_min, y_max, img_height)
        position = (x_norm, y_norm)

        if position in drawn_positions:
            continue

        drawn_positions.add(position)
        color = label_colors[labels[i]]
        image[y_norm, x_norm] = color

    plt.imsave(output_path, image)

def data_files(input_folder, output_folder, num_images=20):
    dataset_count = 0 
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv') or filename.endswith('.dat'):
            file_path = os.path.join(input_folder, filename)
            dataset_name = os.path.splitext(filename)[0]

            dataset_folder = os.path.join(output_folder, dataset_name)
            if not os.path.exists(dataset_folder):
                os.makedirs(dataset_folder)

            X, labels = load_data(file_path)

            for i in range(num_images):
                output_path = os.path.join(dataset_folder, f"{dataset_name}_image_{i + 1}.png")
                create_image(X, labels, output_path, random_state=None)

            dataset_count += 1 


input_folder = '../dataset/data' 
output_folder = 't-SNE_pictures(20)' 

data_files(input_folder, output_folder)
