import pandas as pd
from fuzzywuzzy import fuzz
import snowflake.connector
from dotenv import load_dotenv
import os

load_dotenv('D:\Code\Machine-Learning\GithubRepo\env\.env')

def snowflake_connect():
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )
    return conn

def load_customer_data():
    conn = snowflake_connect()
    query = "SELECT CUSTOMER_ID, LAST_NAME, ADDRESS FROM customers"
    customers = pd.read_sql(query, conn)
    conn.close() 
    return customers

def is_exact_match(row1, row2):
    return row1['LAST_NAME'].strip().lower() == row2['LAST_NAME'].strip().lower() and row1['ADDRESS'].strip().lower() == row2['ADDRESS'].strip().lower()

def is_fuzzy_match(row1, row2, threshold=72):  
    last_name_similarity = fuzz.ratio(row1['LAST_NAME'].strip().lower(), row2['LAST_NAME'].strip().lower())
    address_similarity = fuzz.ratio(row1['ADDRESS'].strip().lower(), row2['ADDRESS'].strip().lower())
    return last_name_similarity >= threshold and address_similarity >= threshold

def assign_cluster_ids(customers):
    customers['CLUSTER_ID'] = -1
    customers['MATCH_TYPE'] = None
    
    current_cluster_id = 1
    
    for i in range(len(customers)):
        if customers.at[i, 'CLUSTER_ID'] == -1:  # If not already assigned a cluster
            customers.at[i, 'CLUSTER_ID'] = current_cluster_id
            customers.at[i, 'MATCH_TYPE'] = 'unique'  # Default match type is 'unique'
            
            for j in range(i + 1, len(customers)):
                if customers.at[j, 'CLUSTER_ID'] == -1: 
                    if is_exact_match(customers.iloc[i], customers.iloc[j]) or is_fuzzy_match(customers.iloc[i], customers.iloc[j]):
                        customers.at[j, 'CLUSTER_ID'] = current_cluster_id
                        if is_exact_match(customers.iloc[i], customers.iloc[j]):
                            customers.at[j, 'MATCH_TYPE'] = 'exact' 
                            customers.at[i, 'MATCH_TYPE'] = 'exact'  
                        else:
                            customers.at[j, 'MATCH_TYPE'] = 'fuzzy' 
                            if customers.at[i, 'MATCH_TYPE'] != 'exact': 
                                customers.at[i, 'MATCH_TYPE'] = 'fuzzy'

            current_cluster_id += 1
    '''
    # Second pass: Fuzzy matching
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            if is_fuzzy_match(customers.iloc[i], customers.iloc[j]) and customers.at[j, 'CLUSTER_ID'] == -1:
                # If there's a fuzzy match, assign the cluster ID of the first matching record
                customers.at[j, 'CLUSTER_ID'] = customers.at[i, 'CLUSTER_ID']
                customers.at[j, 'MATCH_TYPE'] = 'fuzzy'
                if customers.at[i, 'MATCH_TYPE'] == 'unique':  # Update match type of original if necessary
                    customers.at[i, 'MATCH_TYPE'] = 'fuzzy'
    '''
    
    return customers

def save_clustered_data(customers):
    conn = snowflake_connect()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE OR REPLACE TABLE customers_with_clusters (
        CUSTOMER_ID INT,
        LAST_NAME STRING,
        ADDRESS STRING,
        CLUSTER_ID INT,
        MATCH_TYPE STRING
    );
    """)

    insert_query = """
    INSERT INTO customers_with_clusters (CUSTOMER_ID, LAST_NAME, ADDRESS, CLUSTER_ID, MATCH_TYPE)
    VALUES (%s, %s, %s, %s, %s);
    """
    data_to_insert = customers[['CUSTOMER_ID', 'LAST_NAME', 'ADDRESS', 'CLUSTER_ID', 'MATCH_TYPE']].values.tolist()

    cursor.executemany(insert_query, data_to_insert)
    conn.commit()  
    conn.close()

def main():
    customers = load_customer_data()
    clustered_customers = assign_cluster_ids(customers)
    print(clustered_customers)
    #save_clustered_data(clustered_customers)
    #print("Customer data with cluster IDs saved successfully!")

if __name__ == "__main__":
    main()
