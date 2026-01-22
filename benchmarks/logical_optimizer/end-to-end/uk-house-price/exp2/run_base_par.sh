#bash

# max_parallelism range
max_parallelism_range=(1 2 4 32)
sizes_range=(200K 1M 5M)

echo "max_parallelism,size,duration" > base_par.csv

for max_parallelism in ${max_parallelism_range[@]}; do
    echo "Running with max_parallelism: $max_parallelism"
    for size in ${sizes_range[@]}; do
        echo "Size: $size" >> base_par.dat
        # measure the time to execute and write to a file:
        duration=$(time python base_par.py --size $size --cv 3 --max_parallelism $max_parallelism --sleep_time 1 2>&1)
        echo "$max_parallelism,$size,$duration" >> base_par.csv
    done
done