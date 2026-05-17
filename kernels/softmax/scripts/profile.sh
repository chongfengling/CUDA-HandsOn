#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from the script location so the script works from any cwd.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOFTMAX_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SOFTMAX_DIR}/../.." && pwd)"
BUILD_DIR="${SOFTMAX_DIR}/build"

# Defaults can be overridden with environment variables.
NCU_BIN="${NCU_BIN:-ncu}"
BENCH="${BENCH:-${BUILD_DIR}/bench_softmax}"
KERNEL="${KERNEL:-softmax_naive_kernel}"
OUT_DIR="${OUT_DIR:-${SOFTMAX_DIR}/profile_results/$(date +%Y%m%d_%H%M%S)}"
METRICS_FILE="${METRICS_FILE:-${REPO_ROOT}/tools/softmax_metrics.txt}"
LAUNCH_SKIP="${LAUNCH_SKIP:-14}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Profile fixed softmax benchmark shapes with Nsight Compute.

Environment overrides:
  BUILD=1              Build bench_softmax before profiling.
  NCU_BIN=path         Nsight Compute executable. Default: ncu
  BENCH=path           Benchmark binary. Default: ${BUILD_DIR}/bench_softmax
  KERNEL=name          Kernel filter. Default: softmax_naive_kernel
  OUT_DIR=path         Output directory. Default: timestamped profile_results dir
  METRICS_FILE=path    Metric list. Default: ${METRICS_FILE}
  LAUNCH_SKIP=n        Matching launches to skip. Default: ${LAUNCH_SKIP}
  LAUNCH_COUNT=n       Matching launches to profile. Default: ${LAUNCH_COUNT}

Outputs:
  summary.tsv          Per-shape index of generated files.
  *.ncu-rep            Nsight Compute reports.
  *.raw.csv            Vertical CSV: section,name,unit,value.
  *.ncu.log            Nsight Compute logs.
  *.stdout.log         Benchmark stdout.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Fixed benchmark shapes for this profiling pass.
SHAPES=(
  "32768 128"
  "8192 512"
  "8192 1024"
  "4096 4096"
  "1024 8192"
  "64 16384"
)

# Optional one-shot build: BUILD=1 kernels/softmax/scripts/profile.sh
if [[ "${BUILD:-0}" == "1" ]]; then
  cmake -S "${SOFTMAX_DIR}" -B "${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --target bench_softmax -j
fi

if [[ ! -x "${BENCH}" ]]; then
  echo "missing benchmark: ${BENCH}" >&2
  echo "run: BUILD=1 $0" >&2
  exit 1
fi

# Strip comments/blank lines and pass metrics to ncu as a comma-separated list.
METRICS="$(
  sed 's/#.*//; /^[[:space:]]*$/d; s/^[[:space:]]*//; s/[[:space:]]*$//' "${METRICS_FILE}" |
    paste -sd, -
)"

# Convert ncu's wide raw CSV into a stable section/name/unit/value layout.
csv_to_vertical() {
  python3 - "$1" "$2" "$3" <<'PY'
import csv
import sys

src, dst, metrics_file = sys.argv[1], sys.argv[2], sys.argv[3]

metric_order = []
metric_sections = {}
current_section = "metrics"

# Keep metric order and sections exactly as defined in the metrics file.
with open(metrics_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            tag = "# section:"
            if line.lower().startswith(tag):
                current_section = line[len(tag):].strip()
            continue
        metric = line.split("#", 1)[0].strip()
        if metric:
            metric_order.append(metric)
            metric_sections[metric] = current_section

with open(src, newline="") as f:
    rows = list(csv.reader(f))

if len(rows) < 3:
    raise SystemExit(f"unexpected csv format: {src}")

headers, units, values = rows[:3]
# ncu raw CSV stores metric names, units, and values as separate rows.
by_name = {
    name: (unit, value)
    for name, unit, value in zip(headers, units, values)
    if name and value != ""
}

# Keep launch context first, then emit metrics in metrics-file order.
context_names = [
    "ID",
    "Process ID",
    "Process Name",
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
]

with open(dst, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["section", "name", "unit", "value"])

    for name in context_names:
        if name in by_name:
            unit, value = by_name[name]
            writer.writerow(["context", name, unit, value])

    for name in metric_order:
        if name not in by_name:
            continue
        unit, value = by_name[name]
        writer.writerow([metric_sections[name], name, unit, value])
PY
}

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.tsv"
echo -e "M\tN\treport\traw_csv\tlog\tstdout" > "${SUMMARY}"

echo "out: ${OUT_DIR}"
echo "bench: ${BENCH}"
echo "kernel: ${KERNEL}"
echo "launch skip/count: ${LAUNCH_SKIP}/${LAUNCH_COUNT}"

# Profile each shape, export the raw CSV, then rewrite it into the readable layout.
for shape in "${SHAPES[@]}"; do
  read -r M N <<< "${shape}"
  PREFIX="${OUT_DIR}/M${M}_N${N}"
  REPORT="${PREFIX}.ncu-rep"
  WIDE_CSV="${PREFIX}.wide.csv"
  RAW_CSV="${PREFIX}.raw.csv"
  LOG="${PREFIX}.ncu.log"
  STDOUT="${PREFIX}.stdout.log"

  echo "profile M=${M} N=${N}"

  "${NCU_BIN}" \
    --force-overwrite \
    --target-processes application-only \
    --kernel-name-base function \
    --kernel-name "${KERNEL}" \
    --launch-skip "${LAUNCH_SKIP}" \
    --launch-count "${LAUNCH_COUNT}" \
    --disable-extra-suffixes \
    --metrics "${METRICS}" \
    --export "${REPORT}" \
    --log-file "${LOG}" \
    "${BENCH}" "${M}" "${N}" \
    > "${STDOUT}"

  "${NCU_BIN}" \
    --import "${REPORT}" \
    --page raw \
    --csv \
    --log-file "${WIDE_CSV}" \
    >/dev/null

  csv_to_vertical "${WIDE_CSV}" "${RAW_CSV}" "${METRICS_FILE}"
  rm -f "${WIDE_CSV}"

  echo -e "${M}\t${N}\t${REPORT}\t${RAW_CSV}\t${LOG}\t${STDOUT}" >> "${SUMMARY}"
done

echo "done: ${SUMMARY}"
