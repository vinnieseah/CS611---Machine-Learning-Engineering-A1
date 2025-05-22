from pyspark.sql.functions import when

def audit_parquet(spark, paths, name, key_cols=None):
    files = expand(paths)
    if not files:
        print(f"\n--- No parquet files found for {name} at {paths}")
        return
    df = spark.read.parquet(*files)
    total = df.count()
    print(f"\n=== Auditing {name} ===")
    print(f"Total rows: {total:,}\n")

    if key_cols:
        dup_count = total - df.dropDuplicates(key_cols).count()
        print(f"{'Column':30s}{'Nulls':>10s}{'% Null':>10s}{' Dups':>8s}")
        for c in key_cols:
            nulls = df.filter(col(c).isNull()).count()
            pct   = nulls / total * 100
            print(f"{c:30s}{nulls:10,}{pct:10.2f}%{dup_count:8,}")
    else:
        print("  (no key‐column audit)")

    # ─── new: full‐column null audit ───
    # compute null counts for each column
    null_stats = []
    for c in df.columns:
        n = df.filter(col(c).isNull()).count()
        if n > 0:
            null_stats.append((c, n, n/total*100))
    if null_stats:
        print(f"\nColumns with NULL values in {name}:")
        print(f"{'Column':30s}{'Nulls':>10s}{'% Null':>10s}")
        for colname, n, pct in sorted(null_stats, key=lambda x: x[2], reverse=True):
            print(f"{colname:30s}{n:10,}{pct:10.2f}%")
    else:
        print(f"\nNo NULLs found outside key columns in {name}.")
