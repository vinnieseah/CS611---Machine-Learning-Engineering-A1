import os
from pyspark.sql.functions import (
    col, ceil, when, add_months, datediff, trim, regexp_replace
)
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType
)

def process_silver_loan_daily(snapshot_date_str, bronze_dir, silver_dir, spark):
    # prepare arguments
    fp = os.path.join(
        bronze_dir,
        f"bronze_loan_daily_{snapshot_date_str.replace('-','_')}.csv"
    )
    df = spark.read.csv(fp, header=True, inferSchema=True).filter(col("snapshot_date")==snapshot_date_str)
    print("loaded loan_daily:", df.count())

    # enforce schema / data type
    df = df.select(
        col("loan_id"),
        col("Customer_ID").alias("customer_id").cast(StringType()),
        col("loan_start_date").cast(DateType()),
        col("tenure").cast(IntegerType()),
        col("installment_num").cast(IntegerType()),
        col("loan_amt").cast(FloatType()),
        col("due_amt").cast(FloatType()),
        col("paid_amt").cast(FloatType()),
        col("overdue_amt").cast(FloatType()),
        col("balance").cast(FloatType()),
        col("snapshot_date").cast(DateType())
    )

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num"))

    # augment data: add days past due
    df = df.withColumn("installments_missed", ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", when(col("installments_missed") > 0,
                                                add_months(col("snapshot_date"), -col("installments_missed")))
                                   .cast(DateType()))
    df = df.withColumn("dpd", when(col("overdue_amt") > 0.0,
                                   datediff(col("snapshot_date"), col("first_missed_date")))
                                  .otherwise(0).cast(IntegerType()))

    # +: drop exact duplicates
    df = df.dropDuplicates(["loan_id","snapshot_date"])

    print("Silver loan_daily sample:")
    df.show(3, truncate=False)

    # save silver table
    out = os.path.join(silver_dir, f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(out)
    return df


def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    fp = os.path.join(
        bronze_dir,
        f"bronze_feature_clickstream_{snapshot_date_str.replace('-','_')}.csv"
    )
    df = spark.read.csv(fp, header=True, inferSchema=True).filter(col("snapshot_date")==snapshot_date_str)
    print("loaded clickstream:", df.count())

    exprs = [
        col("Customer_ID").alias("customer_id").cast(StringType()),
        col("snapshot_date").cast(DateType())
    ]
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    exprs += [col(c).cast(FloatType()) for c in fe_cols]
    df = df.select(*exprs)

    # +: drop duplicates
    df = df.dropDuplicates(["customer_id","snapshot_date"])

    print("Silver clickstream sample:")
    df.show(3, truncate=False)

    out = os.path.join(silver_dir, f"silver_feature_clickstream_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(out)
    return df


def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    fp = os.path.join(
        bronze_dir,
        f"bronze_features_attributes_{snapshot_date_str.replace('-','_')}.csv"
    )
    df = spark.read.csv(fp, header=True, inferSchema=True).filter(col("snapshot_date")==snapshot_date_str)
    print("loaded attributes:", df.count())

    df = df.select(
        col("Customer_ID").alias("customer_id").cast(StringType()),
        col("Age").cast(IntegerType()).alias("age"),
        col("Occupation").cast(StringType()).alias("occupation"),
        col("snapshot_date").cast(DateType())
    )

    # clamp bad ages
    df = df.withColumn("age", when((col("age")<0)|(col("age")>120), None).otherwise(col("age")))

    # clean odd characters in occupation
    df = df.withColumn("occupation", trim(regexp_replace(col("occupation"), "[^A-Za-z\\s]", " ")))
    df = df.withColumn("occupation", when(trim(col("occupation"))=="", "Unknown").otherwise(col("occupation")))

    # +: drop duplicates
    df = df.dropDuplicates(["customer_id","snapshot_date"])

    print("Silver attributes sample (cleaned):")
    df.show(3, truncate=False)

    out = os.path.join(silver_dir, f"silver_features_attributes_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(out)
    return df


def process_silver_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    fp = os.path.join(
        bronze_dir,
        f"bronze_features_financials_{snapshot_date_str.replace('-','_')}.csv"
    )
    df = spark.read.csv(fp, header=True, inferSchema=True).filter(col("snapshot_date")==snapshot_date_str)
    print("loaded financials:", df.count())

    df = df.select(
        col("Customer_ID").alias("customer_id").cast(StringType()),
        when(col("Type_of_Loan").isNull(), "Unknown").otherwise(trim(col("Type_of_Loan"))).alias("type_of_loan"),
        regexp_replace(col("Annual_Income"), "[^0-9\\.]", "").cast(FloatType()).alias("annual_income"),
        col("Monthly_Inhand_Salary").cast(FloatType()).alias("monthly_inhand_salary"),
        col("Num_Bank_Accounts").cast(IntegerType()).alias("num_bank_accounts"),
        col("Num_Credit_Card").cast(IntegerType()).alias("num_credit_card"),
        col("Interest_Rate").cast(FloatType()).alias("interest_rate"),
        regexp_replace(col("Num_of_Loan"), "[^0-9]", "").cast(IntegerType()).alias("num_of_loan"),
        col("Delay_from_due_date").cast(FloatType()).alias("delay_from_due_date"),
        col("Payment_Behaviour").cast(StringType()).alias("payment_behaviour"),
        col("Monthly_Balance").cast(FloatType()).alias("monthly_balance"),
        col("snapshot_date").cast(DateType())
    )

    # clamp outliers & clean payment_behaviour
    df = (
      df
      .withColumn("interest_rate", when((col("interest_rate")<0)|(col("interest_rate")>100), None).otherwise(col("interest_rate")))
      .withColumn("annual_income",   when((col("annual_income")<0)|(col("annual_income")>1e7), None).otherwise(col("annual_income")))
      .withColumn("num_bank_accounts", when((col("num_bank_accounts")<0)|(col("num_bank_accounts")>100), None).otherwise(col("num_bank_accounts")))
      .withColumn("delay_from_due_date", when(col("delay_from_due_date")<0, None).otherwise(col("delay_from_due_date")))
      .withColumn("monthly_balance", when(col("monthly_balance")<0, None).otherwise(col("monthly_balance")))
    )
    df = df.withColumn("payment_behaviour", trim(regexp_replace(col("payment_behaviour"), "[^A-Za-z\\s]", " ")))
    df = df.withColumn("payment_behaviour", when(trim(col("payment_behaviour"))=="", "Unknown").otherwise(col("payment_behaviour")))

    # +: drop duplicates
    df = df.dropDuplicates(["customer_id","snapshot_date"])

    print("Silver financials sample (clamped & cleaned):")
    df.show(3, truncate=False)

    out = os.path.join(silver_dir, f"silver_features_financials_{snapshot_date_str.replace('-','_')}.parquet")
    df.write.mode("overwrite").parquet(out)
    return df
