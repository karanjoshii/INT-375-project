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

df["population"] = pd.to_numeric(df["population"], errors="coerce")
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["households"] = pd.to_numeric(df["households"], errors="coerce")

df = df[df["area"] > 0]  # Removing zero or negative areas

# Calculating Population Density
df["population_density"] = df["population"] / df["area"]

# Descriptive Statistics
print(df.describe())

# Identifying highest and lowest population density
max_density = df.loc[df["population_density"].idxmax()]
min_density = df.loc[df["population_density"].idxmin()]
print(f"Highest Population Density:\n{max_density}")
print(f"Lowest Population Density:\n{min_density}")

# Debugging available columns before plotting
print("Available columns for plotting:", df.columns.tolist())

# Visualizing Population Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["population"], bins=30, kde=True)
plt.title("Population Distribution")
plt.xlabel("Population")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Household Size vs Population Density
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["households"], y=df["population_density"], alpha=0.6)
plt.title("Household Size vs Population Density")
plt.xlabel("Number of Households")
plt.ylabel("Population Density (people per sq km)")
plt.show()

# Scatter Plot: Population vs Area
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["area"], y=df["population"], alpha=0.5)
plt.title("Population vs. Area")
plt.xlabel("Land Area (sq km)")
plt.ylabel("Population")
plt.show()

# Correlation Heatmap (excluding non-numeric columns)
numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
