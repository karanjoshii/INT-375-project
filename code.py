# India Census Analysis Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
file_path = "C:/Users/karan/Downloads/A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
df = pd.read_excel(file_path, skiprows=4)

# Rename columns
df.columns = [
    "Sr_No", "Region_Code1", "Region_Code2", "Level", "Region_Name", "Area_Type",
    "No_of_Villages", "No_of_Towns", "Uninhabited_Villages", "No_of_Households",
    "Total_Population", "Male_Population", "Female_Population", "Area_sq_km",
    "Population_per_sq_km"
]

# Clean data
df = df[df["Region_Name"].notna() & df["Total_Population"].notna()]
df = df[df["Level"] == "STATE"]
df = df[df["Area_Type"] == "Total"]
df.reset_index(drop=True, inplace=True)

# Derived metrics
df["Household_Size"] = df["Total_Population"] / df["No_of_Households"]
df["Urbanization_Level"] = df["No_of_Towns"] / (df["No_of_Towns"] + df["No_of_Villages"])

# --- Visualizations --- #

sns.set(style="whitegrid")

# [existing plots retained here...]

# 10. Combined Chart: Bar + Line Plot for Population vs Urbanization Level
fig, ax1 = plt.subplots(figsize=(14,6))
color = 'tab:blue'
ax1.set_xlabel('State')
ax1.set_ylabel('Total Population', color=color)
ax1.bar(df['Region_Name'], df['Total_Population'], color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=90)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Urbanization Level', color=color)
ax2.plot(df['Region_Name'], df['Urbanization_Level'], color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)
plt.title("Total Population and Urbanization Level by State")
plt.tight_layout()
plt.show()

# 11. Combined Chart: Scatter + Regression line for Area vs Population Density
plt.figure(figsize=(10,6))
sns.regplot(data=df, x='Area_sq_km', y='Population_per_sq_km', scatter_kws={'s': 50, 'alpha': 0.7})
plt.title("Area vs Population Density with Regression Line")
plt.xlabel("Area (sq.km)")
plt.ylabel("Population Density")
plt.tight_layout()
plt.show()

# 12. Combined Chart: Violin + Swarm Plot for Household Size
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x='Area_Type', y='Household_Size', inner=None, color="lightblue")
sns.swarmplot(data=df, x='Area_Type', y='Household_Size', color="black", alpha=0.6)
plt.title("Violin + Swarm Plot of Household Size by Area Type")
plt.tight_layout()
plt.show()

# 13. Bar Plot: Top 10 States by Population
plt.figure(figsize=(12,6))
top_pop_states = df.sort_values('Total_Population', ascending=False).head(10)
sns.barplot(data=top_pop_states, x='Region_Name', y='Total_Population', hue='Region_Name', palette='Blues_d', legend=False)
plt.title('Top 10 States by Total Population')
plt.xlabel('State')
plt.ylabel('Total Population')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 14. Scatter Plot: Urbanization Level vs Population Density
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Urbanization_Level', y='Population_per_sq_km', hue='Region_Name', palette='viridis', s=100, alpha=0.8)
plt.title('Urbanization Level vs Population Density')
plt.xlabel('Urbanization Level')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()

# 15. Box Plot: Distribution of Household Size by Region
plt.figure(figsize=(14,6))
sns.boxplot(data=df, x='Region_Name', y='Household_Size', hue='Region_Name', palette='Pastel1', legend=False)
plt.xticks(rotation=90)
plt.title('Box Plot of Household Size by State')
plt.xlabel('State')
plt.ylabel('Household Size')
plt.tight_layout()
plt.show()


# 16. Scatter Plot: Area vs Household Size with Different Colors
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Area_sq_km', y='Household_Size', hue='Region_Name', palette='tab20', s=100, alpha=0.85)
plt.title('Scatter Plot: Area vs Household Size by State')
plt.xlabel('Area (sq.km)')
plt.ylabel('Household Size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 17. Distribution Plot: Total Population
plt.figure(figsize=(10,6))
sns.histplot(df['Total_Population'], kde=True, color='blue')
plt.title('Distribution of Total Population Across Regions')
plt.xlabel('Total Population')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 18. Heatmap: Correlation between Variables
correlation_matrix = df[['Total_Population', 'Male_Population', 'Female_Population',
                         'Household_Size', 'Urbanization_Level', 'Population_per_sq_km', 'Area_sq_km']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 19. Pie Chart: Proportion of Urban vs Rural Population
urban_population = df['Urbanization_Level'].mean()
rural_population = 1 - urban_population
labels = ['Urban', 'Rural']
sizes = [urban_population, rural_population]
plt.figure(figsize=(7,7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#99ff99'])
plt.title('Urban vs Rural Population Proportion')
plt.tight_layout()
plt.show()



# 20. Facet Grid: Population and Household Size by Region Type
g = sns.FacetGrid(df, col="Area_Type", height=6, aspect=1.5)
g.map(sns.scatterplot, "Total_Population", "Household_Size", alpha=.7)
g.set_axis_labels("Total Population", "Household Size")
g.set_titles("{col_name} Area")
plt.tight_layout()
plt.show()

# 21. Pair Plot: Relationship Between Multiple Variables
sns.pairplot(df[['Total_Population', 'Household_Size', 'Urbanization_Level', 'Population_per_sq_km', 'Area_sq_km']])
plt.suptitle('Pair Plot of Population and Household Size Relationships', y=1.02)
plt.tight_layout()
plt.show()


#22 Box Plot for Outliers in Total Population
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region_Name', y='Total_Population', palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Box Plot of Total Population by Region with Outliers')
plt.xlabel('Region Name')
plt.ylabel('Total Population')
plt.tight_layout()
plt.show()


# --- Additional Visualizations --- #

# Create top_density_states before printing
top_density_states = df[['Region_Name', 'Population_per_sq_km']].sort_values(by='Population_per_sq_km', ascending=False).head(5)

# 1. Bar Plot: Bottom 5 States by Population Density
bottom_density_states = df[['Region_Name', 'Population_per_sq_km']].sort_values(by='Population_per_sq_km', ascending=True).head(5)
plt.figure(figsize=(10,6))
sns.barplot(data=bottom_density_states, x='Region_Name', y='Population_per_sq_km', hue='Region_Name', palette='coolwarm', legend=False)
plt.title('Bottom 5 States by Population Density')
plt.xlabel('State')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()

# 2. Box Plot: Household Size by Region Type
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='Area_Type', y='Household_Size', hue='Area_Type', palette='Set2', legend=False)
plt.title('Box Plot of Household Size by Area Type')
plt.xlabel('Area Type')
plt.ylabel('Household Size')
plt.tight_layout()
plt.show()

# 3. Violin Plot: Distribution of Population Density
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x='Area_Type', y='Population_per_sq_km', hue='Area_Type', palette='coolwarm', legend=False)
plt.title('Violin Plot of Population Density by Area Type')
plt.xlabel('Area Type')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()

# 4. Box Plot of Total Population by Region (fixed)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region_Name', y='Total_Population', hue='Region_Name', palette='coolwarm', legend=False)
plt.xticks(rotation=90)
plt.title('Box Plot of Total Population by Region with Outliers')
plt.xlabel('Region Name')
plt.ylabel('Total Population')
plt.tight_layout()
plt.show()

# 5. Print Summary Tables
print("Top 5 States by Population Density:")
print(top_density_states)
