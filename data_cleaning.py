import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("visualizations", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset Shape:", df.shape)

# Missing Values
for col in df.select_dtypes(include=['int64', 'float64']):
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=['object']):
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove Duplicates
df = df.drop_duplicates()

# Save cleaned data
df.to_csv("cleaned_dataset.csv", index=False)

# Numeric Columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Histogram
plt.figure(figsize=(8,5))
df[num_cols[0]].hist()
plt.title("Data Distribution")
plt.savefig("visualizations/histogram.png")
plt.close()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("visualizations/heatmap.png")
plt.close()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(data=df[num_cols])
plt.title("Box Plot")
plt.savefig("visualizations/boxplot.png")
plt.close()

print("Project Completed Successfully")