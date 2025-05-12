import pandas as pd
from datasets import load_dataset
import numpy as np

# Load the dataset
dataset = load_dataset("danavery/urbansound8K")

# Convert to pandas DataFrame for easier analysis
df = pd.DataFrame(dataset['train'])

# Print available columns
print("\nAvailable columns:")
print(df.columns.tolist())

# Get class distribution
class_distribution = df['classID'].value_counts().sort_index()
print("\nClass Distribution:")
for class_id, count in class_distribution.items():
    class_name = df[df['classID'] == class_id]['class'].iloc[0]
    print(f"Class {class_id} ({class_name}): {count} examples")

# Group data once for efficiency
grouped = df.groupby('classID')

# Mean length (duration) per class
# print("\nMean Length (seconds) by Class:")
# for class_id, group in grouped:
#     class_name = group['class'].iloc[0]
#     mean_length = group['audio'].apply(lambda x: x['duration']).mean()
#     print(f"Class {class_id} ({class_name}): {mean_length:.2f} seconds")

# Mean salience per class
print("\nMean Salience by Class:")
for class_id, group in grouped:
    class_name = group['class'].iloc[0]
    mean_salience = group['salience'].mean()
    print(f"Class {class_id} ({class_name}): {mean_salience:.2f}")

# Overall statistics
print("\nOverall Statistics:")
print(f"Total number of examples: {len(df)}")
print(f"Number of unique classes: {df['classID'].nunique()}")
# print(f"Mean length across all classes: {df['audio'].apply(lambda x: x['duration']).mean():.2f} seconds")
print(f"Mean salience across all classes: {df['salience'].mean():.2f}")

