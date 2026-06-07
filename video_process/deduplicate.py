import argparse
import pandas as pd

# Example usage:
# python deduplicate.py all file1.csv file2.csv file3.csv -o result.csv
# python deduplicate.py relative -a A.csv -b B.csv -o A_final.csv

def dedup_all(input_files, output_file, column="photo_id"):
    """Merge multiple CSV files and output unique IDs"""
    all_ids = []
    total_rows = 0

    for file in input_files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        if column not in df.columns:
            print(f"Column {column} not found in file {file}")
            continue

        num_rows = len(df)
        total_rows += num_rows
        print(f"{file}: {num_rows} rows")

        all_ids.extend(df[column].astype(str).tolist())

    # Deduplicate
    unique_ids = sorted(set(all_ids))

    # Stats
    total_duplicates = len(all_ids) - len(unique_ids)
    print(f"\nSummary:")
    print(f"  Total rows read (across all files): {total_rows}")
    print(f"  Total entries collected: {len(all_ids)}")
    print(f"  Duplicates found: {total_duplicates}")
    print(f"  Unique {column} values: {len(unique_ids)}")

    # Save result
    unique_df = pd.DataFrame({column: unique_ids})
    unique_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nDone! Unique values saved to {output_file}")


def dedup_relative(A_file, B_file, output_file, column="photo_id"):
    """Deduplicate A, then remove any IDs that also exist in B"""
    try:
        df_A = pd.read_csv(A_file)
    except Exception as e:
        print(f"Error reading A file {A_file}: {e}")
        return

    try:
        df_B = pd.read_csv(B_file)
    except Exception as e:
        print(f"Error reading B file {B_file}: {e}")
        return

    if column not in df_A.columns:
        print(f"Column {column} not found in A file {A_file}")
        return
    if column not in df_B.columns:
        print(f"Column {column} not found in B file {B_file}")
        return

    # Step 1: deduplicate A itself
    A_unique = set(df_A[column].astype(str).tolist())
    print(f"A file {A_file}: {len(df_A)} rows -> {len(A_unique)} unique after self-deduplication")

    # Step 2: get B IDs
    B_ids = set(df_B[column].astype(str).tolist())
    print(f"B file {B_file}: {len(df_B)} rows -> {len(B_ids)} unique IDs")

    # Step 3: subtract B from A
    A_final = sorted(A_unique - B_ids)
    print(f"After removing B IDs: {len(A_final)} unique IDs remain in A_final")

    # Save result
    final_df = pd.DataFrame({column: A_final})
    final_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nDone! A_final saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV deduplication tool")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Mode 1: dedup_all
    parser_all = subparsers.add_parser("all", help="Merge multiple CSV files and deduplicate")
    parser_all.add_argument("input_files", nargs="+", help="Input CSV file paths (one or more)")
    parser_all.add_argument("-o", "--output", default="deduplicated.csv", help="Output CSV file name")
    parser_all.add_argument("-c", "--column", default="photo_id", help="Column name to deduplicate, default is photo_id")

    # Mode 2: dedup_relative
    parser_rel = subparsers.add_parser("relative", help="Deduplicate A and remove IDs found in B")
    parser_rel.add_argument("-a", "--afile", required=True, help="CSV file A (to be deduplicated)")
    parser_rel.add_argument("-b", "--bfile", required=True, help="CSV file B (reference)")
    parser_rel.add_argument("-o", "--output", default="A_final.csv", help="Output CSV file name")
    parser_rel.add_argument("-c", "--column", default="photo_id", help="Column name to deduplicate, default is photo_id")

    args = parser.parse_args()

    if args.mode == "all":
        dedup_all(args.input_files, args.output, args.column)
    elif args.mode == "relative":
        dedup_relative(args.afile, args.bfile, args.output, args.column)
