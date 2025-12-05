import pandas as pd

input_file = "meta_features.xlsx"
output_file = "cleaned_meta_features.xlsx"

df = pd.read_excel(input_file)

df_cleaned = df.dropna(axis=1)

df_cleaned.to_excel(output_file, index=False)

