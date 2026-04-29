#!/bin/bash

# download dataset from kaggle
curl -L -o uk-housing-prices-paid.zip\
  https://www.kaggle.com/api/v1/datasets/download/hm-land-registry/uk-housing-prices-paid

unzip uk-housing-prices-paid.zip -d tmp
mkdir -p input
mv tmp/* input/
rm -rf tmp
rm uk-housing-prices-paid.zip

# downsample for testing:
head -100000  input/price_paid_records.csv > input/price_paid_records_100K.csv
head -1000000 input/price_paid_records.csv > input/price_paid_records_1M.csv
