import pandas as pd
dataset_directory = "/Users/ryanlouie/Downloads/LinkedIn_Dataset.pcl" #Change this according to your directory
dataset = pd.read_pickle(dataset_directory)
print(dataset.head())
print(dataset.tail())

print(dataset.info())

# Generate statistical summary
print(dataset.describe())