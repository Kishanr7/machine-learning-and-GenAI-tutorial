import pandas as pd
from fuzzywuzzy import fuzz
import hashlib

# Function to generate a unique hash based on last name and address
def generate_cluster_id(last_name, address):
    unique_string = f"{last_name}_{address}"
    return hashlib.md5(unique_string.encode()).hexdigest()

# Reading the CSV file into a dataframe
df = pd.read_csv('D:\Code\Machine-Learning\GithubRepo\machine-learning-and-GenAI-tutorial\FuzzyMatching\customers.csv')

# Initially, assign a unique cluster ID based on last_name and address for each record
df['CLUSTER_ID'] = df.apply(lambda row: generate_cluster_id(row['last_name'], row['address']), axis=1)

# Merge the dataframe on last_name to find exact matched records
merged_df = df.merge(df, on='last_name', suffixes=('_1', '_2'))

# Filter out self-comparisons
merged_df = merged_df[merged_df['customer_id_1'] != merged_df['customer_id_2']]

# Create a new column for address match (either exact or fuzzy match with threshold of 80)
merged_df['address_match'] = merged_df.apply(
    lambda x: (x['address_1'] == x['address_2']) or (fuzz.ratio(x['address_1'], x['address_2']) >= 80),
    axis=1
)

# Filter to keep only matched records
matching_df = merged_df[merged_df['address_match']]

# Iterate through each row of matching df and assign the same cluster ID for matching records
for _, row in matching_df.iterrows():
    # Get the cluster ID of the first matched customer
    cluster_id_1 = df.loc[df['customer_id'] == row['customer_id_1'], 'CLUSTER_ID'].values[0]
    cluster_id_2 = df.loc[df['customer_id'] == row['customer_id_2'], 'CLUSTER_ID'].values[0]

    # Assign the same cluster ID for both customers if they match
    if cluster_id_1 != cluster_id_2:
        # Assign the cluster ID of the first customer to the second one
        df.loc[df['customer_id'] == row['customer_id_2'], 'CLUSTER_ID'] = cluster_id_1

# Sorting values by customer_id
df = df.sort_values(by='customer_id')

# Print the final result
print(df[['last_name', 'address', 'CLUSTER_ID']])
