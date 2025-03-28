import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/karan/Downloads/A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
df = pd.read_excel(file_path)

# Display available columns
print("Columns in dataset:", df.columns.tolist())

# Identifying the correct header row and relevant columns
df = df.iloc[5:]  # Adjusting to skip initial non-data rows
df.columns = ["state_ut", "district", "sub_district", "villages", "towns", "households", "population", "area", "extra1", "extra2", "extra3", "extra4", "extra5", "extra6", "extra7"]

# Selecting only relevant columns
df = df[["state_ut", "district", "sub_district", "villages", "towns", "households", "population", "area"]]

# Dropping unnecessary rows and handling missing values
df.dropna(subset=["population", "area", "households"], inplace=True)
