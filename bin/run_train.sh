#!/bin/bash

source /etc/profile

cd ../src

declare -a exp_ids=("001" "002" "003" "004" "005" "006" "007" "008" "009")

for exp_id in "${exp_ids[@]}"
do
  echo "Running inference on $exp_id ..."
  python3 run_train.py \
          --conf='../conf/config.yaml' \
          --exp_id=$exp_id
done
