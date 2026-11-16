#!/usr/bin/env bash
set -euo pipefail

destination=${1:?Usage: $0 DATA_DIRECTORY}
mkdir -p "$destination"

if [[ ! -f "$destination/price_paid_records.csv" ]]; then
    uv run --no-project --with kagglehub==1.0.3 python - "$destination" <<'PY'
import sys
import kagglehub

kagglehub.dataset_download("hm-land-registry/uk-housing-prices-paid",
                           path="price_paid_records.csv", output_dir=sys.argv[1])
PY
fi

# Include the header in each sample.
head -n 100001 "$destination/price_paid_records.csv" > "$destination/price_paid_records_100K.csv"
head -n 200001 "$destination/price_paid_records.csv" > "$destination/price_paid_records_200K.csv"
head -n 1000001 "$destination/price_paid_records.csv" > "$destination/price_paid_records_1M.csv"
