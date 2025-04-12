# India Census Analysis Script

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
file_path = "C:/Users/karan/Downloads/A-1_NO_OF_VILLAGES_TOWNS_HOUSEHOLDS_POPULATION_AND_AREA.xlsx"
df = pd.read_excel(file_path, skiprows=4)
print(df.info())
print(df.describe())

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

# 1. Combined Chart: Bar + Line Plot for Population vs Urbanization Level
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

# 2. Combined Chart: Scatter + Regression line for Area vs Population Density
plt.figure(figsize=(10,6))
sns.regplot(data=df, x='Area_sq_km', y='Population_per_sq_km', scatter_kws={'s': 50, 'alpha': 0.7})
plt.title("Area vs Population Density with Regression Line")
plt.xlabel("Area (sq.km)")
plt.ylabel("Population Density")
plt.tight_layout()
plt.show()

# 3. Combined Chart: Violin + Swarm Plot for Household Size
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x='Area_Type', y='Household_Size', inner=None, color="lightblue")
sns.swarmplot(data=df, x='Area_Type', y='Household_Size', color="black", alpha=0.6)
plt.title("Violin + Swarm Plot of Household Size by Area Type")
plt.tight_layout()
plt.show()

# 4. Bar Plot: Top 10 States by Population
plt.figure(figsize=(12,6))
top_pop_states = df.sort_values('Total_Population', ascending=False).head(10)
sns.barplot(data=top_pop_states, x='Region_Name', y='Total_Population', hue='Region_Name', palette='Blues_d', legend=False)
plt.title('Top 10 States by Total Population')
plt.xlabel('State')
plt.ylabel('Total Population')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 5. Scatter Plot: Urbanization Level vs Population Density
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Urbanization_Level', y='Population_per_sq_km', hue='Region_Name', palette='viridis', s=100, alpha=0.8)
plt.title('Urbanization Level vs Population Density')
plt.xlabel('Urbanization Level')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()

# 6. Box Plot: Distribution of Household Size by Region
plt.figure(figsize=(14,6))
sns.boxplot(data=df, x='Region_Name', y='Household_Size', hue='Region_Name', palette='Pastel1', legend=False)
plt.xticks(rotation=90)
plt.title('Box Plot of Household Size by State')
plt.xlabel('State')
plt.ylabel('Household Size')
plt.tight_layout()
plt.show()


# 7. Scatter Plot: Area vs Household Size with Different Colors
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Area_sq_km', y='Household_Size', hue='Region_Name', palette='tab20', s=100, alpha=0.85)
plt.title('Scatter Plot: Area vs Household Size by State')
plt.xlabel('Area (sq.km)')
plt.ylabel('Household Size')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 8. Distribution Plot: Total Population
plt.figure(figsize=(10,6))
sns.histplot(df['Total_Population'], kde=True, color='blue')
plt.title('Distribution of Total Population Across Regions')
plt.xlabel('Total Population')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 9. Heatmap: Correlation between Variables
correlation_matrix = df[['Total_Population', 'Male_Population', 'Female_Population',
                         'Household_Size', 'Urbanization_Level', 'Population_per_sq_km', 'Area_sq_km']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 10. Pie Chart: Proportion of Urban vs Rural Population
urban_population = df['Urbanization_Level'].mean()
rural_population = 1 - urban_population
labels = ['Urban', 'Rural']
sizes = [urban_population, rural_population]
plt.figure(figsize=(7,7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#99ff99'])
plt.title('Urban vs Rural Population Proportion')
plt.tight_layout()
plt.show()



# 11. Facet Grid: Population and Household Size by Region Type
g = sns.FacetGrid(df, col="Area_Type", height=6, aspect=1.5)
g.map(sns.scatterplot, "Total_Population", "Household_Size", alpha=.7)
g.set_axis_labels("Total Population", "Household Size")
g.set_titles("{col_name} Area")
plt.tight_layout()
plt.show()

# 12. Pair Plot: Relationship Between Multiple Variables
sns.pairplot(df[['Total_Population', 'Household_Size', 'Urbanization_Level', 'Population_per_sq_km', 'Area_sq_km']])
plt.suptitle('Pair Plot of Population and Household Size Relationships', y=1.02)
plt.tight_layout()
plt.show()


#13 Box Plot for Outliers in Total Population
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region_Name', y='Total_Population', palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Box Plot of Total Population by Region with Outliers')
plt.xlabel('Region Name')
plt.ylabel('Total Population')
plt.tight_layout()
plt.show()



# Create top_density_states before printing
top_density_states = df[['Region_Name', 'Population_per_sq_km']].sort_values(by='Population_per_sq_km', ascending=False).head(5)

# 14. Bar Plot: Bottom 5 States by Population Density
bottom_density_states = df[['Region_Name', 'Population_per_sq_km']].sort_values(by='Population_per_sq_km', ascending=True).head(5)
plt.figure(figsize=(10,6))
sns.barplot(data=bottom_density_states, x='Region_Name', y='Population_per_sq_km', hue='Region_Name', palette='coolwarm', legend=False)
plt.title('Bottom 5 States by Population Density')
plt.xlabel('State')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()

# 15. Box Plot: Household Size by Region Type
plt.figure(figsize=(12,6))
sns.boxplot(data=df, x='Area_Type', y='Household_Size', hue='Area_Type', palette='Set2', legend=False)
plt.title('Box Plot of Household Size by Area Type')
plt.xlabel('Area Type')
plt.ylabel('Household Size')
plt.tight_layout()
plt.show()

# 16. Violin Plot: Distribution of Population Density
plt.figure(figsize=(12,6))
sns.violinplot(data=df, x='Area_Type', y='Population_per_sq_km', hue='Area_Type', palette='coolwarm', legend=False)
plt.title('Violin Plot of Population Density by Area Type')
plt.xlabel('Area Type')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()

# 17. Box Plot of Total Population by Region (fixed)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Region_Name', y='Total_Population', hue='Region_Name', palette='coolwarm', legend=False)
plt.xticks(rotation=90)
plt.title('Box Plot of Total Population by Region with Outliers')
plt.xlabel('Region Name')
plt.ylabel('Total Population')
plt.tight_layout()
plt.show()

# 18. Print Summary Tables
print("Top 5 States by Population Density:")
print(top_density_states)

#19. Horizontal Bar Chart: States by Number of Towns
plt.figure(figsize=(12,8))
towns_sorted = df.sort_values(by='No_of_Towns', ascending=True)
sns.barplot(data=towns_sorted, x='No_of_Towns', y='Region_Name', palette='crest')
plt.title('Number of Towns by State')
plt.xlabel('Number of Towns')
plt.ylabel('State')
plt.tight_layout()
plt.show()

#20. Bubble Chart: Area vs Population with Size by Household Size
plt.figure(figsize=(12,8))
sns.scatterplot(
    data=df,
    x='Area_sq_km',
    y='Total_Population',
    size='Household_Size',
    hue='Region_Name',
    palette='tab20',
    sizes=(100, 1000),
    alpha=0.7,
    legend=False
)
plt.title('Area vs Total Population (Bubble Size: Household Size)')
plt.xlabel('Area (sq.km)')
plt.ylabel('Total Population')
plt.tight_layout()
plt.show()


#21. Lollipop Chart: Population Density
sorted_df = df.sort_values(by='Population_per_sq_km', ascending=False)
plt.figure(figsize=(14,6))
plt.hlines(y=sorted_df['Region_Name'], xmin=0, xmax=sorted_df['Population_per_sq_km'], color='skyblue')
plt.plot(sorted_df['Population_per_sq_km'], sorted_df['Region_Name'], "o")
plt.title('Lollipop Chart: Population Density by State')
plt.xlabel('Population per sq.km')
plt.ylabel('State')
plt.tight_layout()
plt.show()


#22. Donut Chart: Distribution of Total Population by Top 5 States
top5 = df.sort_values('Total_Population', ascending=False).head(5)
sizes = top5['Total_Population']
labels = top5['Region_Name']
colors = sns.color_palette('pastel')[0:5]
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
plt.title('Total Population Share of Top 5 States')
plt.tight_layout()
plt.show()


#23. Count Plot: Number of States per Area Type
plt.figure(figsize=(6,5))
sns.countplot(data=df, x='Area_Type', palette='Set3')
plt.title('Count of States by Area Type')
plt.tight_layout()
plt.show()


#24. Histogram: Household Size Distribution
plt.figure(figsize=(10,6))
sns.histplot(df['Household_Size'], bins=20, kde=True, color='teal')
plt.title('Distribution of Household Size')
plt.xlabel('Household Size')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


#25. Bar Plot: Gender Ratio (Females per 1000 Males) by State
df['Gender_Ratio'] = (df['Female_Population'] / df['Male_Population']) * 1000
plt.figure(figsize=(14,6))
sns.barplot(data=df.sort_values('Gender_Ratio', ascending=False), x='Region_Name', y='Gender_Ratio', palette='magma')
plt.xticks(rotation=90)
plt.title('Gender Ratio (Females per 1000 Males) by State')
plt.tight_layout()
plt.show()


#26. KDE Plot: Comparison of Male vs Female Population
plt.figure(figsize=(10,6))
sns.kdeplot(df['Male_Population'], label='Male', fill=True)
sns.kdeplot(df['Female_Population'], label='Female', fill=True)
plt.title('KDE Plot: Male vs Female Population Distribution')
plt.xlabel('Population')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()


#27. Bar Plot: State-wise Urbanization Level
plt.figure(figsize=(14,6))
df_sorted = df.sort_values('Urbanization_Level', ascending=False)
sns.barplot(data=df_sorted, x='Region_Name', y='Urbanization_Level', palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Urbanization Level by State')
plt.ylabel('Urbanization Level (Proportion)')
plt.tight_layout()
plt.show()


#28. Heatmap: Top 10 Most Densely Populated States
top10_density = df.sort_values(by='Population_per_sq_km', ascending=False).head(10)
pivot = top10_density.pivot_table(index='Region_Name', values='Population_per_sq_km')
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, cmap='Reds', fmt='.0f')
plt.title('Heatmap: Top 10 Most Densely Populated States')
plt.tight_layout()
plt.show()


#29. Joint Plot: Area vs Total Population
sns.jointplot(data=df, x='Area_sq_km', y='Total_Population', kind='hex', height=8, color='green')
plt.suptitle('Joint Plot: Area vs Total Population', y=1.02)
plt.show()

#30. Line Plot: Total Population vs Gender Counts
plt.figure(figsize=(12,6))
df_sorted = df.sort_values('Total_Population')
plt.plot(df_sorted['Region_Name'], df_sorted['Total_Population'], label='Total Population', color='blue')
plt.plot(df_sorted['Region_Name'], df_sorted['Male_Population'], label='Male', color='skyblue')
plt.plot(df_sorted['Region_Name'], df_sorted['Female_Population'], label='Female', color='lightpink')
plt.xticks(rotation=90)
plt.title('Population Comparison by Gender')
plt.legend()
plt.tight_layout()
plt.show()

#31. Heatmap: Gender Distribution by State
gender_df = df[['Region_Name', 'Male_Population', 'Female_Population']].set_index('Region_Name')
gender_df = gender_df.div(gender_df.sum(axis=1), axis=0)  # Normalize
plt.figure(figsize=(12,6))
sns.heatmap(gender_df, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Proportion of Male and Female Population by State')
plt.tight_layout()
plt.show()

#32. Line Plot: Top 10 States by Urbanization Level
top_urban = df.sort_values('Urbanization_Level', ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.lineplot(data=top_urban, x='Region_Name', y='Urbanization_Level', marker='o', color='darkorange')
plt.title('Top 10 States by Urbanization Level')
plt.xlabel('State')
plt.ylabel('Urbanization Level')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#33. Radar Chart: Comparison of Key Indicators for Top 5 States by Population
from math import pi

# Prepare data
top5_states = df.sort_values('Total_Population', ascending=False).head(5)
indicators = ['Household_Size', 'Urbanization_Level', 'Population_per_sq_km']
categories = indicators + [indicators[0]]

plt.figure(figsize=(8,8))

for i, row in top5_states.iterrows():
    values = row[indicators].tolist()
    values += values[:1]  # loop back
    angles = [n / float(len(indicators)) * 2 * pi for n in range(len(indicators))]
    angles += angles[:1]
    plt.polar(angles, values, marker='o', label=row['Region_Name'])

plt.xticks([n / float(len(indicators)) * 2 * pi for n in range(len(indicators))], indicators)
plt.title('Radar Chart: Top 5 States by Key Indicators')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()


#34. Bar Plot: Top 5 States with Most Uninhabited Villages
top_uninhabited = df.sort_values('Uninhabited_Villages', ascending=False).head(5)
plt.figure(figsize=(10,6))
sns.barplot(data=top_uninhabited, x='Region_Name', y='Uninhabited_Villages', palette='flare')
plt.title('Top 5 States with Most Uninhabited Villages')
plt.xlabel('State')
plt.ylabel('Number of Uninhabited Villages')
plt.tight_layout()
plt.show()

#35. Strip Plot: Distribution of Population Density
plt.figure(figsize=(12,6))
sns.stripplot(data=df, x='Region_Name', y='Population_per_sq_km', palette='Spectral', size=8)
plt.xticks(rotation=90)
plt.title('Strip Plot: Population Density Distribution by State')
plt.xlabel('State')
plt.ylabel('Population per sq.km')
plt.tight_layout()
plt.show()





