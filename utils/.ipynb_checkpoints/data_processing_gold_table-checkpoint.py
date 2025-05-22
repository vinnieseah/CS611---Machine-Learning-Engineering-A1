import os
import glob                                    # +: to glob all silver parquet files
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col, split, array_contains, greatest,
    trim, regexp_replace
)
from pyspark.sql.window import Window
from pyspark.sql.types  import IntegerType

def process_gold_feature_store(
    snapshot_date_str, silver_dir, gold_feat_dir, spark
):
    os.makedirs(gold_feat_dir, exist_ok=True)
    s = snapshot_date_str.replace("-", "_")

    # load this month's silver loan + click
    df_loan  = spark.read.parquet(f"{silver_dir}/silver_loan_daily_{s}.parquet")
    df_click = spark.read.parquet(f"{silver_dir}/silver_feature_clickstream_{s}.parquet")

    # +: load full history of attrs & financials for as-of joins
    attr_paths = glob.glob(f"{silver_dir}/silver_features_attributes_*.parquet")      # +: all dt
    df_attr_all = spark.read.parquet(*attr_paths)                                     # +: union
    fin_paths  = glob.glob(f"{silver_dir}/silver_features_financials_*.parquet")      # +: all dt
    df_fin_all  = spark.read.parquet(*fin_paths)                                      # +: union

    # 1) as-of join for attributes
    df_attr2 = df_attr_all.select(
        col("customer_id"),
        col("snapshot_date").alias("attr_date"),
        col("age"),
        col("occupation")
    )
    loan_attr = (
        df_loan.alias("loan")
          .join(
            df_attr2.alias("attr"),
            (col("loan.customer_id")==col("attr.customer_id")) &
            (col("attr.attr_date") <= col("loan.snapshot_date")),
            how="left"
          )
          .withColumn(
            "rn",
            F.row_number().over(
              Window.partitionBy("loan.loan_id","loan.snapshot_date")
                    .orderBy(F.desc("attr_date"))
            )
          )
          .filter(col("rn")==1)
          .drop("rn")
    )
    base1 = loan_attr.select(
        "loan.*",
        col("attr.age"),
        col("attr.occupation")
    )

    # 2) as-of join for financials
    df_fin2 = df_fin_all.select(
        col("customer_id"),
        col("snapshot_date").alias("fin_date"),
        col("annual_income"),
        col("monthly_inhand_salary"),
        col("num_bank_accounts"),
        col("num_credit_card"),
        col("interest_rate"),
        col("num_of_loan"),
        col("type_of_loan"),
        col("delay_from_due_date"),
        col("payment_behaviour"),
        col("monthly_balance")
    )
    loan_attr_fin = (
        base1.alias("base")
          .join(
            df_fin2.alias("fin"),
            (col("base.customer_id")==col("fin.customer_id")) &
            (col("fin.fin_date") <= col("base.snapshot_date")),
            how="left"
          )
          .withColumn(
            "rn",
            F.row_number().over(
              Window.partitionBy("base.loan_id","base.snapshot_date")
                    .orderBy(F.desc("fin_date"))
            )
          )
          .filter(col("rn")==1)
          .drop("rn")
    )
    base2 = loan_attr_fin.select(
        "base.*",
        col("fin.annual_income"),
        col("fin.monthly_inhand_salary"),
        col("fin.num_bank_accounts"),
        col("fin.num_credit_card"),
        col("fin.interest_rate"),
        col("fin.num_of_loan"),
        col("fin.type_of_loan"),
        col("fin.delay_from_due_date"),
        col("fin.payment_behaviour"),
        col("fin.monthly_balance")
    )

    # 3) left-join clickstream + zero-fill
    joined = base2.alias("bf").join(
        df_click.alias("clk"),
        on=["customer_id","snapshot_date"],
        how="left"
    )
    fe_cols = [c for c in df_click.columns if c.startswith("fe_")]
    joined = joined.fillna(0, subset=fe_cols)

    # 4) select and rename (no more clk.fe_ ambiguity)
    sel = [
        col("loan_id"), col("customer_id"), col("snapshot_date"),
        col("loan_start_date"), col("tenure"), col("installment_num"),
        col("loan_amt"), col("due_amt"), col("paid_amt"),
        col("overdue_amt"), col("balance"), col("mob"),
        col("installments_missed"), col("dpd")
    ] + [col(c) for c in fe_cols] + [
        col("age"), col("occupation"),
        col("annual_income"), col("monthly_inhand_salary"),
        col("num_bank_accounts"), col("num_credit_card"),
        col("interest_rate"), col("num_of_loan"),
        col("type_of_loan"), col("delay_from_due_date"),
        col("payment_behaviour"), col("monthly_balance")
    ]
    feat = joined.select(*sel)

    # 5) one-hot encode loan types
    types = ["Auto Loan","Payday Loan","Credit-Builder Loan",
             "Home Equity Loan","Student Loan","Mortgage Loan","Unknown"]
    feat = feat.withColumn("loan_type_list",
                           split(regexp_replace(col("type_of_loan"), " and ", ","), ","))
    for lt in types:
        key = lt.replace(" ","_").replace("-","_")
        feat = feat.withColumn(
            f"loan_type_{key}",
            array_contains(col("loan_type_list"), lt).cast(IntegerType())
        )
    feat = feat.drop("loan_type_list")

    # 6) age bins
    feat = feat.withColumn(
        "age_bin",
        F.when(col("age")<25, "<25")
         .when(col("age")<35, "25_34")
         .when(col("age")<50, "35_49")
         .otherwise("50_plus")
    )
    bins = ["<25","25_34","35_49","50_plus"]
    for b in bins:
        safe = b.replace("<","lt_").replace("+","_plus")
        feat = feat.withColumn(f"age_bin_{safe}", (col("age_bin")==b).cast(IntegerType()))
    feat = feat.drop("age_bin")

    # 7) interactions
    feat = feat.withColumn("pay_ratio",    col("paid_amt")/greatest(col("due_amt"),    F.lit(1))) \
               .withColumn("bal_ratio",    col("balance")/greatest(col("loan_amt"),    F.lit(1))) \
               .withColumn("ovd_ratio",    col("overdue_amt")/greatest(col("due_amt"),F.lit(1))) \
               .withColumn("cum_paid",     col("installment_num")*col("due_amt")) \
               .withColumn("late_per_miss",col("dpd")/greatest(col("installments_missed"),F.lit(1)))
    bases = ["pay_ratio","bal_ratio","ovd_ratio","cum_paid","late_per_miss"]
    for f1 in bases:
        for f2 in bases:
            feat = feat.withColumn(f"{f1}_x_{f2}", col(f1)*col(f2))

    out = os.path.join(gold_feat_dir, f"gold_feature_store_{s}.parquet")
    feat.write.mode("overwrite").parquet(out)
    print("Saved gold features:", out)
    return feat


def process_gold_label_store(
    snapshot_date_str, silver_dir, gold_lbl_dir, spark, dpd, mob
):
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col
    from pyspark.sql.types     import IntegerType, StringType

    os.makedirs(gold_lbl_dir, exist_ok=True)
    s = snapshot_date_str.replace("-", "_")

    df = spark.read.parquet(f"{silver_dir}/silver_loan_daily_{s}.parquet")
    df = df.filter(col("mob")==mob)
    df = df.withColumn("label", F.when(col("dpd")>=dpd, 1).otherwise(0).cast(IntegerType())) \
           .withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id","customer_id","label","label_def","snapshot_date")

    out = os.path.join(gold_lbl_dir, f"gold_label_store_{s}.parquet")
    df.write.mode("overwrite").parquet(out)
    print("Saved gold labels:", out)
    return df
