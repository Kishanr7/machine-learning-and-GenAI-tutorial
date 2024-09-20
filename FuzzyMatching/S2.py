import pandas as pd
from fuzzywuzzy import fuzz

df = pd.read_csv('D:\Code\Machine-Learning\GithubRepo\machine-learning-and-GenAI-tutorial\FuzzyMatching\customers.csv')

df['ROW_ID'] = range(len(df))

def is_fuzzy_match(addr1, addr2, threshold=80):
    return fuzz.ratio(addr1, addr2) >= threshold

df['CLUSTER_ID'] = df['ROW_ID'] + 1

merged_df = df.merge(df, on='last_name', suffixes=('_1', '_2'))

merged_df = merged_df[merged_df['ROW_ID_1'] != merged_df['ROW_ID_2']]

merged_df['address_match'] = merged_df.apply(
    lambda x: (x['address_1'] == x['address_2']) or is_fuzzy_match(x['address_1'], x['address_2']),
    axis=1
)

matching_df = merged_df[merged_df['address_match']]

for _, row in matching_df.iterrows():
    min_cluster_id = min(df.loc[df['ROW_ID'] == row['ROW_ID_1'], 'CLUSTER_ID'].values[0],
                         df.loc[df['ROW_ID'] == row['ROW_ID_2'], 'CLUSTER_ID'].values[0])
    
    df.loc[df['ROW_ID'] == row['ROW_ID_1'], 'CLUSTER_ID'] = min_cluster_id
    df.loc[df['ROW_ID'] == row['ROW_ID_2'], 'CLUSTER_ID'] = min_cluster_id

df = df.sort_values(by='ROW_ID')

print(df[['last_name', 'address', 'CLUSTER_ID']])
# df.to_csv('path_to_output.csv', index=False)