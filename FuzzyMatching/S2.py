#This script is written to match columns with either exact match or a fuzzy match with a certain threshold defined.
import pandas as pd
from fuzzywuzzy import fuzz

#Reading the csv file into a dataframe. I have used a local version, we can swap this with AWS api to use csv from s3 directly.
df = pd.read_csv('D:\Code\Machine-Learning\GithubRepo\machine-learning-and-GenAI-tutorial\FuzzyMatching\customers.csv')

#assigning the first cluster id as 1
df['CLUSTER_ID'] = range(1, len(df) + 1)

#merging the dataframe on last_name to find out exact matched records and assigning two suffixes to separate both columns
merged_df = df.merge(df, on='last_name', suffixes=('_1', '_2'))

# this filter outs records with only unique self comparison
merged_df = merged_df[merged_df['customer_id_1'] != merged_df['customer_id_2']]

'''Creates a new column called address_match which will store a boolean value
it applies the logic written within the lambda function across rows/columns of the dataframe.
Lambda logic has two shots - 1. exact match and fuzzy match with a ratio of 80.
The function is applied to each row of the DataFrame.
'''
merged_df['address_match'] = merged_df.apply(
    lambda x: (x['address_1'] == x['address_2']) or (fuzz.ratio(x['address_1'], x['address_2']) >= 80),
    axis=1
)

#filters out merged_df to keep rows only where address match is true resulting dataframe matching_Df only contains matched records
matching_df = merged_df[merged_df['address_match']]

'''iterates through each row of matching df.
logic is to find out minimum of the cluster id between two records and check the same for both customers.
if both of them matches they will be assigned the same cluster id which will be the minimum of the two.
'''
for _, row in matching_df.iterrows():
    min_cluster_id = min(df.loc[df['customer_id'] == row['customer_id_1'], 'CLUSTER_ID'].values[0],
                         df.loc[df['customer_id'] == row['customer_id_2'], 'CLUSTER_ID'].values[0])
    
    df.loc[df['customer_id'] == row['customer_id_1'], 'CLUSTER_ID'] = min_cluster_id
    df.loc[df['customer_id'] == row['customer_id_2'], 'CLUSTER_ID'] = min_cluster_id

#sorting values by customer id
df = df.sort_values(by='customer_id')

print(df[['last_name', 'address', 'CLUSTER_ID']])
# df.to_csv('path_to_output.csv', index=False)