import argparse
import csv
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(r"^(?P<algo>.+)_M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)\.raw\.csv$")


def latest_result_dir(base_dir):
    if not base_dir.exists():
        return None
    dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    return max(dirs, default=None, key=lambda path: path.name)


def read_summary(target_dir):
    summary = target_dir / "summary.tsv"
    if not summary.exists():
        return []

    runs = []
    with summary.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("raw_csv"):
                runs.append(row)
    return runs


def runs_from_filenames(target_dir):
    runs = []
    for raw_csv in sorted(target_dir.glob("*.raw.csv")):
        match = FILENAME_RE.match(raw_csv.name)
        if not match:
            continue
        run = match.groupdict()
        run["raw_csv"] = str(raw_csv)
        run["stdout"] = str(raw_csv.with_name(raw_csv.name.replace(".raw.csv", ".stdout.log")))
        runs.append(run)
    return runs


def metric_label(row):
    name = str(row["name"])
    section = str(row["section"])
    unit = row.get("unit", "")
    label = f"{section}/{name}"
    if pd.notna(unit) and str(unit) != "":
        label = f"{label} ({unit})"
    return label


def read_metrics(raw_csv):
    df = pd.read_csv(raw_csv)
    required = {"section", "name", "unit", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

    df = df[df["section"] != "context"].copy()
    metrics = OrderedDict()
    duplicate_counts = defaultdict(int)

    for _, row in df.iterrows():
        label = metric_label(row)
        duplicate_counts[label] += 1
        if duplicate_counts[label] > 1:
            label = f"{label} #{duplicate_counts[label]}"
        metrics[label] = row["value"]

    return metrics


def parse_stdout(stdout):
    path = Path(stdout)
    if not path.exists():
        return {}

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    row = rows[-1]
    return {
        "Time_ms": row.get("Time_ms", ""),
        "GFLOPS": row.get("GFLOPS", ""),
    }


def run_label(run):
    return f"{run['algo']}_M{run['M']}_N{run['N']}_K{run['K']}"


def shape_label(run):
    return f"M{run['M']}_N{run['N']}_K{run['K']}"


def normalize_run(row):
    return {
        "algo": row.get("algo") or row.get("Algorithm") or row.get("ALGORITHM"),
        "kernel": row.get("kernel", ""),
        "M": row.get("M"),
        "N": row.get("N"),
        "K": row.get("K"),
        "raw_csv": row.get("raw_csv"),
        "stdout": row.get("stdout", ""),
    }


def collect_runs(target_dir):
    runs = [normalize_run(row) for row in read_summary(target_dir)]
    if not runs:
        runs = [normalize_run(row) for row in runs_from_filenames(target_dir)]

    valid_runs = []
    for run in runs:
        if not all([run["algo"], run["M"], run["N"], run["K"], run["raw_csv"]]):
            print(f"Skipping incomplete run: {run}")
            continue
        raw_csv = Path(run["raw_csv"])
        if not raw_csv.is_absolute() and not raw_csv.exists():
            raw_csv = target_dir / raw_csv
        run["raw_csv"] = str(raw_csv)

        stdout = run.get("stdout", "")
        if stdout:
            stdout_path = Path(stdout)
            if not stdout_path.is_absolute() and not stdout_path.exists():
                stdout_path = target_dir / stdout_path
            run["stdout"] = str(stdout_path)

        if not raw_csv.exists():
            print(f"Skipping missing CSV: {run['raw_csv']}")
            continue
        valid_runs.append(run)
    return valid_runs


def write_grouped_tables(target_dir, merged_df, runs):
    by_shape = defaultdict(list)
    by_algo = defaultdict(list)

    for run in runs:
        label = run_label(run)
        by_shape[shape_label(run)].append(label)
        by_algo[run["algo"]].append(label)

    shape_dir = target_dir / "by_shape"
    algo_dir = target_dir / "by_algorithm"
    shape_dir.mkdir(exist_ok=True)
    algo_dir.mkdir(exist_ok=True)

    for shape, columns in by_shape.items():
        if len(columns) > 1:
            merged_df[columns].to_csv(shape_dir / f"{shape}.csv")

    for algo, columns in by_algo.items():
        if len(columns) > 1:
            merged_df[columns].to_csv(algo_dir / f"{algo}.csv")


def merge_csv_files(target_dir):
    runs = collect_runs(target_dir)
    if not runs:
        print("No GEMM .raw.csv files found.")
        return None

    columns = OrderedDict()
    run_rows = []

    for run in runs:
        label = run_label(run)
        try:
            columns[label] = read_metrics(run["raw_csv"])
            stdout_metrics = parse_stdout(run.get("stdout", ""))
            run_rows.append({**run, "label": label, **stdout_metrics})
        except Exception as exc:
            print(f"Error processing {run['raw_csv']}: {exc}")

    if not columns:
        print("No valid data extracted.")
        return None

    merged_df = pd.DataFrame(columns)
    output_csv = target_dir / "merged_metrics.csv"
    merged_df.to_csv(output_csv)
    print(f"Saved merged metrics to {output_csv}")

    runs_csv = target_dir / "runs.csv"
    pd.DataFrame(run_rows).to_csv(runs_csv, index=False)
    print(f"Saved run index to {runs_csv}")

    write_grouped_tables(target_dir, merged_df, runs)
    return merged_df


def main():
    parser = argparse.ArgumentParser(description="Analyze GEMM Nsight Compute profile results")
    parser.add_argument("--dir", type=Path, help="Directory containing profile results")
    args = parser.parse_args()

    target_dir = args.dir
    if target_dir is None:
        target_dir = latest_result_dir(Path("kernels/gemm/profile_results"))
    if target_dir is None:
        print("No profile results found.")
        return

    print(f"Analyzing directory: {target_dir}")
    merge_csv_files(target_dir)


if __name__ == "__main__":
    main()
