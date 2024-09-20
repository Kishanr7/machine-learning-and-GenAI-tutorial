from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, monotonically_increasing_id
from pyspark.sql.functions import levenshtein, expr

# Initialize Spark session
spark = SparkSession.builder.appName("DataMatching").getOrCreate()

# Load customer data from CSV
df = spark.read.csv("D:\Code\Machine-Learning\GithubRepo\machine-learning-and-GenAI-tutorial\FuzzyMatching\customers.csv", header=True, inferSchema=True)

# Create an ID column to track original row
df = df.withColumn("ROW_ID", monotonically_increasing_id())

# Self-join on the exact match for last_name and address threshold match
threshold = 0.8

# Calculate the similarity score using Levenshtein distance
similarity_condition = (
    (levenshtein(col("df1.address"), col("df2.address")) / 
     expr("greatest(length(df1.address), length(df2.address))")) <= (1 - threshold)
)

# Self-join with an exact match on last_name and either an exact or threshold-based match on address
joined_df = df.alias("df1").join(
    df.alias("df2"),
    (col("df1.last_name") == col("df2.last_name")) &
    ((col("df1.address") == col("df2.address")) | similarity_condition)
)

# Assign cluster IDs based on ROW_ID
cluster_df = joined_df.withColumn(
    "CLUSTER_ID",
    when(col("df1.ROW_ID") < col("df2.ROW_ID"), col("df1.ROW_ID"))
    .otherwise(col("df2.ROW_ID"))
)

# Select relevant columns
result_df = cluster_df.select(
    "df1.ROW_ID", "df1.last_name", "df1.address", "CLUSTER_ID"
).distinct()

# Show result or save to file
result_df.show()  # Show the result
# result_df.write.csv("path_to_output.csv", header=True)  # Save if needed
