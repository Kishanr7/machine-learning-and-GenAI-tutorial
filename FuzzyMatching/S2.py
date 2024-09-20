import pandas as pd
from fuzzywuzzy import fuzz

df = pd.read_csv('D:\Code\Machine-Learning\GithubRepo\machine-learning-and-GenAI-tutorial\FuzzyMatching\customers.csv')

df['CLUSTER_ID'] = range(1, len(df) + 1)

merged_df = df.merge(df, on='last_name', suffixes=('_1', '_2'))

merged_df = merged_df[merged_df['customer_id_1'] != merged_df['customer_id_2']]

merged_df['address_match'] = merged_df.apply(
    lambda x: (x['address_1'] == x['address_2']) or (fuzz.ratio(x['address_1'], x['address_2']) >= 80),
    axis=1
)

matching_df = merged_df[merged_df['address_match']]

for _, row in matching_df.iterrows():
    min_cluster_id = min(df.loc[df['customer_id'] == row['customer_id_1'], 'CLUSTER_ID'].values[0],
                         df.loc[df['customer_id'] == row['customer_id_2'], 'CLUSTER_ID'].values[0])
    
    df.loc[df['customer_id'] == row['customer_id_1'], 'CLUSTER_ID'] = min_cluster_id
    df.loc[df['customer_id'] == row['customer_id_2'], 'CLUSTER_ID'] = min_cluster_id

df = df.sort_values(by='customer_id')

print(df[['last_name', 'address', 'CLUSTER_ID']])
# df.to_csv('path_to_output.csv', index=False)