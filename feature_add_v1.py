# Adding features to upgrade the Multi class model v1 
# features added: PP to BC, PP to S1, PP to S2

import pandas as pd

# Load your data
df = pd.read_excel("eternal_hist.xlsx")  # or pd.read_excel("your_data.xlsx")

# ==============================
# CREATE NEW FEATURES
# ==============================
df['PP_to_BC_pct'] = abs(df['PP'] - df['BC']) / df['PP'] * 100
df['PP_to_S1_pct'] = abs(df['PP'] - df['S1']) / df['PP'] * 100
df['PP_to_S2_pct'] = abs(df['PP'] - df['S2']) / df['PP'] * 100

dec1_columns = ['PP_to_BC_pct','PP_to_S1_pct', 'PP_to_S2_pct']
df[dec1_columns] = df[dec1_columns].round(2)

# Preview new features
# print(df[['PP_to_BC_pct','PP_to_S1_pct','PP_to_S2_pct']].head())

# ==============================
# SAVE DATASET TO EXCEL
# ==============================
df.to_excel("eternal_hist_ft.xlsx", index=False)
print("Dataset with new features saved as 'reliance_features.xlsx'.")
