import os
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import LabelEncoder

input_folder = "../dataset/data"
output_file = "meta_features.xlsx"

all_meta_features = []

for filename in os.listdir(input_folder):
    if filename.endswith(".csv") or filename.endswith(".dat"):
        file_path = os.path.join(input_folder, filename)

        with open(file_path, 'r') as f:
            first_line = f.readline()
            delimiter = ',' if ',' in first_line else '\\s+'

        try:
            df = pd.read_csv(file_path, sep=delimiter, engine="python", on_bad_lines='skip', header=None)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        X, y = X.values, y.values

        mfe = MFE(groups=["general", "statistical", "info-theory", "model-based", "landmarking"])
        mfe.fit(X, y)
        ft, ft_vals = mfe.extract()

        meta_features_dict = {"Dataset": filename}
        meta_features_dict.update(dict(zip(ft, ft_vals)))
        all_meta_features.append(meta_features_dict)

        print(f"Processed: {filename}")

meta_features_df = pd.DataFrame(all_meta_features)
meta_features_df.to_excel(output_file, index=False)

print(f"Meta-features extraction complete. Results saved to {output_file}")
