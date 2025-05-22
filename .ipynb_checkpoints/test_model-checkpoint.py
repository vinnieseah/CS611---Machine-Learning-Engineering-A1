import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature        import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation     import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

def main():
    spark = (
      SparkSession
        .builder
        .appName("ModelTest")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df_feat = spark.read.parquet("datamart/gold/features/*.parquet")
    df_lbl  = spark.read.parquet("datamart/gold/label_store/*.parquet")
    df      = df_feat.join(df_lbl,
                           ["loan_id","customer_id","snapshot_date"],
                           "inner")

    # drop loan-derived & interactions
    leak = [
      "loan_start_date","tenure","installment_num","loan_amt","due_amt",
      "paid_amt","overdue_amt","balance","mob","installments_missed",
      "dpd","pay_ratio","bal_ratio","ovd_ratio","cum_paid","late_per_miss"
    ]
    for c in df.columns:
        if any(c.startswith(p) for p in leak) or "_x_" in c:
            df = df.drop(c)

    # pick numeric + label
    numeric_types = {"int","bigint","double","float"}
    feature_cols = [
      n for n,t in df.dtypes
      if n not in {"loan_id","customer_id","snapshot_date","label","label_def"}
         and t in numeric_types
    ]
    print("Using features:", feature_cols)

    df = df.na.fill(0, subset=feature_cols)
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="error"
    )
    df_ml = assembler.transform(df).select("snapshot_date","features","label")

    months = sorted(
      df_ml.select("snapshot_date")
           .distinct()
           .rdd.flatMap(lambda r: r)
           .collect()
    )
    cutoff = months[int(len(months)*0.8)]
    train = df_ml.filter(col("snapshot_date") <= cutoff)
    test  = df_ml.filter(col("snapshot_date") >  cutoff)
    print(f"Train rows: {train.count()}, Test rows: {test.count()} (cutoff {cutoff})")

    lr    = LogisticRegression(maxIter=10, featuresCol="features", labelCol="label")
    model = lr.fit(train)
    pred  = model.transform(test)

    auc = BinaryClassificationEvaluator(
        metricName="areaUnderROC", labelCol="label"
    ).evaluate(pred)
    acc = MulticlassClassificationEvaluator(
        metricName="accuracy", labelCol="label"
    ).evaluate(pred)
    print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}")

    spark.stop()

if __name__=="__main__":
    main()
