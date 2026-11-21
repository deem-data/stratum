#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/results/countvectorizer_phases.csv"
N_JOBS_LIST=(1 2 4 8)

rm -f "${OUT}"

for n_jobs in "${N_JOBS_LIST[@]}"; do
    echo "=== n_jobs=${n_jobs} ==="
    python3 "${HERE}/countvectorizer_phases.py" --n-jobs "${n_jobs}" --out "${OUT}"
done

python3 "${HERE}/plot_countvectorizer_phases.py"
