#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: $0 NUMA THREADS [--offset N] DATA.csv [--modes ...]" >&2
    exit 2
fi
numa=$1
threads=$2
shift 2
offset=0
if [[ $1 == --offset ]]; then
    offset=${2:?Missing core offset}
    shift 2
fi
data=${1:?Missing input CSV}
shift

if [[ ! $numa =~ ^(0|[1-9][0-9]*)$ || ! $threads =~ ^[1-9][0-9]*$ ||
      ! $offset =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "NUMA and offset must be nonnegative integers; THREADS must be positive." >&2
    exit 2
fi

# Select one logical CPU per physical core, skipping its SMT sibling.
mapfile -t cpus < <(lscpu --online -p=CPU,CORE,SOCKET,NODE |
    awk -F, -v node="$numa" '!/^#/ && $4 == node && !seen[$3 ":" $2]++ {print $1}')
if (( threads > ${#cpus[@]} || offset > ${#cpus[@]} - threads )); then
    echo "NUMA node $numa has ${#cpus[@]} physical cores; requested $threads at offset $offset." >&2
    exit 2
fi
selected=("${cpus[@]:offset:threads}")
cpu_list=$(IFS=,; echo "${selected[*]}")
output="${data%.csv}.numa${numa}.offset${offset}.threads${threads}.timings.csv"

exec numactl --physcpubind="$cpu_list" --membind="$numa" \
    uv run --no-sync python "$(dirname -- "${BASH_SOURCE[0]}")/benchmark.py" \
    "$data" "$@" --threads "$threads" --output "$output"
