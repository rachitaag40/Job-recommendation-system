# This is a temporary file to understand our data
import pandas as pd

# Read the CSV file
df = pd.read_csv("data/freelancer_earnings - freelancer_earnings_vs_skillstack_dataset.csv")

# Explore it step by step
print("First 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nUnique categories:")
print(df['category'].unique())

