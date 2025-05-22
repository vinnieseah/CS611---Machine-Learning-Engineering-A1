import os, glob
from datetime import datetime
import pyspark
from pyspark.sql.functions import col

import utils.data_processing_bronze_table as bronze_mod
import utils.data_processing_silver_table as silver_mod
import utils.data_processing_gold_table   as gold_mod

# Initialize Spark
spark = (
    pyspark.sql.SparkSession
      .builder
      .appName("MedallionPipeline")
      .master("local[*]")
      .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# Config
start_date_str = "2023-01-01"
end_date_str   = "2024-12-01"
dpd            = 30
mob            = 6

def generate_month_starts(start, end):
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end,   "%Y-%m-%d")
    dates = []
    cur = datetime(sd.year, sd.month, 1)
    while cur <= ed:
        dates.append(cur.strftime("%Y-%m-%d"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1)
        else:
            cur = datetime(cur.year, cur.month + 1, 1)
    return dates

dates = generate_month_starts(start_date_str, end_date_str)

# Bronze layer
bronze_dir = "datamart/bronze"
os.makedirs(bronze_dir, exist_ok=True)
for dt in dates:
    bronze_mod.process_bronze_table(
        dt, "data/lms_loan_daily.csv",      bronze_dir, "bronze_loan_daily", spark
    )
    bronze_mod.process_bronze_table(
        dt, "data/feature_clickstream.csv", bronze_dir, "bronze_feature_clickstream", spark
    )
    bronze_mod.process_bronze_table(
        dt, "data/features_attributes.csv", bronze_dir, "bronze_features_attributes", spark
    )
    bronze_mod.process_bronze_table(
        dt, "data/features_financials.csv", bronze_dir, "bronze_features_financials", spark
    )

# Silver layer
silver_dir = "datamart/silver"
os.makedirs(silver_dir, exist_ok=True)
for dt in dates:
    silver_mod.process_silver_loan_daily(
        dt, bronze_dir, silver_dir, spark
    )
    silver_mod.process_silver_clickstream(
        dt, bronze_dir, silver_dir, spark
    )
    silver_mod.process_silver_attributes(
        dt, bronze_dir, silver_dir, spark
    )
    silver_mod.process_silver_financials(
        dt, bronze_dir, silver_dir, spark
    )

# Gold layer
gold_feat_dir = "datamart/gold/features"
gold_lbl_dir  = "datamart/gold/label_store"
os.makedirs(gold_feat_dir, exist_ok=True)
os.makedirs(gold_lbl_dir,  exist_ok=True)

for dt in dates:
    gold_mod.process_gold_feature_store(
        dt, silver_dir, gold_feat_dir, spark
    )
    gold_mod.process_gold_label_store(
        dt, silver_dir, gold_lbl_dir, spark, dpd=dpd, mob=mob
    )

# Final check
feat = spark.read.parquet(*glob.glob(f"{gold_feat_dir}/*.parquet"))
lbl  = spark.read.parquet(*glob.glob(f"{gold_lbl_dir}/*.parquet"))
df   = feat.join(lbl, ["loan_id","customer_id","snapshot_date"], "inner")
print("Final training set rows:", df.count())
df.show(5, truncate=False)

spark.stop()
