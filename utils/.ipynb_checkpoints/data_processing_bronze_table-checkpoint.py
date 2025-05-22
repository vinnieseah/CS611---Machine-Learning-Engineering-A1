import os
from datetime import datetime
from pyspark.sql.functions import col

def process_bronze_table(
    snapshot_date_str,
    source_csv_path,
    bronze_directory,
    table_name,
    spark
):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # load data
    df = (
        spark.read
             .csv(source_csv_path, header=True, inferSchema=True)
             .filter(col("snapshot_date") == snapshot_date_str)
    )
    print(f"{table_name} {snapshot_date_str} row count:", df.count())

    # save bronze table
    os.makedirs(bronze_directory, exist_ok=True)
    partition = f"{table_name}_{snapshot_date_str.replace('-','_')}.csv"
    out_path  = os.path.join(bronze_directory, partition)
    df.toPandas().to_csv(out_path, index=False)
    print("saved to:", out_path)

    return df
