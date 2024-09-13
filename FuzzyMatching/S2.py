import pandas as pd
from fuzzywuzzy import fuzz
import snowflake.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def snowflake_connect():
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )
    return conn

def load_customer_data():
    conn = snowflake_connect()
    query = "SELECT CUSTOMER_ID, LAST_NAME, ADDRESS FROM customers"
    customers = pd.read_sql(query, conn)
    conn.close()  # Close the connection after fetching the data
    return customers

def is_exact_match(row1, row2):
    return (row1['LAST_NAME'].strip().lower() == row2['LAST_NAME'].strip().lower() and
            row1['ADDRESS'].strip().lower() == row2['ADDRESS'].strip().lower())

def is_fuzzy_match(row1, row2, threshold=75):
    last_name_similarity = fuzz.ratio(row1['LAST_NAME'].strip().lower(), row2['LAST_NAME'].strip().lower())
    address_similarity = fuzz.ratio(row1['ADDRESS'].strip().lower(), row2['ADDRESS'].strip().lower())
    return last_name_similarity >= threshold and address_similarity >= threshold

import networkx as nx

def assign_cluster_ids(df):
    G = nx.Graph()

    # Add nodes
    for i, row in df.iterrows():
        G.add_node(i, data=row)

    # Add edges for exact and fuzzy matches
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i != j:
                if is_exact_match(row1, row2) or is_fuzzy_match(row1, row2):
                    G.add_edge(i, j)

    # Find connected components
    clusters = list(nx.connected_components(G))

    # Assign cluster IDs
    cluster_ids = [None] * len(df)
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            cluster_ids[node] = cluster_id

    df['ClusterID'] = cluster_ids
    return df

df = assign_cluster_ids(load_customer_data())
print(df)