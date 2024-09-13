import pandas as pd
from fuzzywuzzy import fuzz
import snowflake.connector
from dotenv import load_dotenv
import os

load_dotenv()

# Define your connection parameters
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
    conn.close()  # Close the connection after fetching the data
    return customers

def is_exact_match(row1, row2):
    return row1['LAST_NAME'].strip().lower() == row2['LAST_NAME'].strip().lower() and row1['ADDRESS'].strip().lower() == row2['ADDRESS'].strip().lower()

def is_fuzzy_match(row1, row2, threshold=72):  # Increase threshold to 85 for stricter matches
    last_name_similarity = fuzz.ratio(row1['LAST_NAME'].strip().lower(), row2['LAST_NAME'].strip().lower())
    address_similarity = fuzz.ratio(row1['ADDRESS'].strip().lower(), row2['ADDRESS'].strip().lower())
    return last_name_similarity >= threshold and address_similarity >= threshold

# Step 4: Assign Cluster IDs based on exact and fuzzy matches
def assign_cluster_ids(customers):
    # Initialize cluster IDs with -1 (unassigned)
    customers['CLUSTER_ID'] = -1
    customers['MATCH_TYPE'] = None
    
    current_cluster_id = 1
    
    # Iterate over each row to check for exact and fuzzy matches
    for i in range(len(customers)):
        if customers.at[i, 'CLUSTER_ID'] == -1:  # If not already assigned a cluster
            customers.at[i, 'CLUSTER_ID'] = current_cluster_id
            customers.at[i, 'MATCH_TYPE'] = 'unique'  # Default match type is 'unique'
            
            for j in range(i + 1, len(customers)):
                if customers.at[j, 'CLUSTER_ID'] == -1:  # Only unassigned rows
                    if is_exact_match(customers.iloc[i], customers.iloc[j]) or is_fuzzy_match(customers.iloc[i], customers.iloc[j]):
                        customers.at[j, 'CLUSTER_ID'] = current_cluster_id
                        if is_exact_match(customers.iloc[i], customers.iloc[j]):
                            customers.at[j, 'MATCH_TYPE'] = 'exact'  # Exact match found
                            customers.at[i, 'MATCH_TYPE'] = 'exact'  # Update first record's match type if exact
                        else:
                            customers.at[j, 'MATCH_TYPE'] = 'fuzzy'  # Fuzzy match found
                            if customers.at[i, 'MATCH_TYPE'] != 'exact':  # Update the first record if it's not already an exact match
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

# Step 5: Save Clustered Data back to Snowflake
def save_clustered_data(customers):
    conn = snowflake_connect()
    # Write DataFrame back to Snowflake
    cursor = conn.cursor()
    # Create a table if not exists for the results
    cursor.execute("""
    CREATE OR REPLACE TABLE customers_with_clusters (
        CUSTOMER_ID INT,
        LAST_NAME STRING,
        ADDRESS STRING,
        CLUSTER_ID INT,
        MATCH_TYPE STRING
    );
    """)

    # Prepare data for insertion
    insert_query = """
    INSERT INTO customers_with_clusters (CUSTOMER_ID, LAST_NAME, ADDRESS, CLUSTER_ID, MATCH_TYPE)
    VALUES (%s, %s, %s, %s, %s);
    """
    data_to_insert = customers[['CUSTOMER_ID', 'LAST_NAME', 'ADDRESS', 'CLUSTER_ID', 'MATCH_TYPE']].values.tolist()

    # Execute insertion
    cursor.executemany(insert_query, data_to_insert)
    conn.commit()  # Commit the transaction
    conn.close()

# Step 6: Main workflow
def main():
    customers = load_customer_data()
    clustered_customers = assign_cluster_ids(customers)
    print(clustered_customers)
    #save_clustered_data(clustered_customers)
    #print("Customer data with cluster IDs saved successfully!")

# Run the solution
if __name__ == "__main__":
    main()
