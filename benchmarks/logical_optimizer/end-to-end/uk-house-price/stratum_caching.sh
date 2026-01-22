#!/usr/bin/env bash

# define range
start=200000
end=1000000
step=200000

for n in $(seq $start $step $end); do
  echo $n
  slice="input/price_paid_records_head_${n}.csv"

  head -n "$n" input/price_paid_records.csv > "$slice"

  rm -rf ~/.stratum/cache/*

  # fill caches
  python3 stratum_level1.py $slice > /dev/null 2>&1

  secs=$(
    { /usr/bin/time -p bash -c "
        python3 stratum_level2.py  $slice
      " ; } 2>&1 | awk '/^real /{print $2}'
  )

  echo "$n $secs" >> stratum_caching.dat

  rm -f "$slice"
done